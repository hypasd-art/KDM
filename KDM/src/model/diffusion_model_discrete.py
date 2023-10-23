"""
Part of code is copied and adapted from https://arxiv.org/abs/2209.14734
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import igraph


from model.train_metrics import TrainLossDiscrete
import utils

import model.diffusion_utils as diffusion_utils
from model.transformer_model import GraphTransformer

class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar).cuda()
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar[t_int.long()]


class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

class DiscreteDenoisingDiffusion(nn.Module):
    def __init__(self, args, dataset_infos, depth_sampler):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.args = args
        self.name = args.name
        self.model_dtype = torch.float32
        self.T = args.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.args.lambda_train)

        self.depth_sampler = depth_sampler
        self.device = torch.device("cuda")

        self.model = GraphTransformer(args = self.args,
                                      n_layers=args.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=args.hidden_mlp_dims,
                                      hidden_dims=args.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(args.diffusion_noise_schedule,
                                                              timesteps=args.diffusion_steps)

        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                            y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.number_chain_steps = args.number_chain_steps


    def set_gpu(self, data=None, node_mask=None):
        data.X = data.X.cuda()
        data.E = data.E.cuda()
        node_mask = node_mask.cuda()
        return data, node_mask
    
    def loss(self, data):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        depth = data.depth
        depth = depth.reshape(-1, self.args.max_n)
        dense_data, node_mask = self.set_gpu(dense_data, node_mask)
        dense_data = dense_data.mask(node_mask)

        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        if self.args.use_pos:
            pred = self.forward(noisy_data, node_mask, depth)
        else:
            pred = self.forward(noisy_data, node_mask)

        loss, loss_X, loss_E = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y)


        return loss, loss_X, loss_E

    def sample_test(self, type_c, sample_num, min_n = 25, max_n=35) -> None:
        samples_num_to_generate = sample_num

        process_list  = []
        process_samples = []
        samples = []
        while samples_num_to_generate > 0:
            print(f'Samples left to generate: {len(samples)}/' f'{samples_num_to_generate}')
            to_generate = min(samples_num_to_generate, 500)
            final_list, process_list = self.sample_batch(type_c, to_generate, num_nodes=None, min_n = min_n, max_n = max_n)
            samples.extend(final_list)
            process_samples.extend(process_list)
            samples_num_to_generate -= to_generate

        print("saving graph")
        dict_new = self._load_event_ontology()
        gs = []
        for idx, item in enumerate(samples):
            g = igraph.Graph(directed=True)
           
            n = item[0].shape[0]
            g.add_vertices(n)
            for i, node_type in enumerate(item[0]):
                g.vs[i]['type'] = dict_new[node_type.item()]
            for i, edge_list in enumerate(item[1]):
                for j, edge in enumerate(edge_list):
                    if edge == 1:
                        g.add_edge(i,j)
            g = self._reorder_graph(g)
            gs.append(g)
        with open(self.args.res_dir[type_c] + '/predict_igraph.pkl', 'wb') as handle:
            pickle.dump(gs, handle)     
        print("save in " + self.args.res_dir[type_c])
        print("Done.")


        if self.args.save_process_pic:
            print("generate process")
            print(len(process_list))
            for ids, sampl in enumerate(process_samples):
                gs = []
                for idx, item in enumerate(sampl):
                    print(idx)
                    g = igraph.Graph(directed=True)
                
                    n = item[0].shape[0]
                    g.add_vertices(n)
                    for i, node_type in enumerate(item[0]):
                        g.vs[i]['type'] = dict_new[node_type.item()]
                    for i, edge_list in enumerate(item[1]):
                        for j, edge in enumerate(edge_list):
                            if edge == 1:
                                g.add_edge(i,j)
                    gs.append(g)
                    # print(idx)
                with open(self.args.res_dir[type_c] +  '/process_predict_igraph'+str(ids)+'.pkl', 'wb') as handle:
                    pickle.dump([gs,sampl], handle)     
                print("save in " + self.args.res_dir[type_c] +  '/process_predict_igraph'+str(ids)+'.pkl')
                print("Done.")

  
    def sample_test_single(self, sample_num, min_n = 25, max_n = 35) -> None:
        samples_num_to_generate = sample_num

        process_list  = []
        process_samples = []
        samples = []
        while samples_num_to_generate > 0:
            print(f'Samples left to generate: {len(samples)}/' f'{samples_num_to_generate}')
            to_generate = min(samples_num_to_generate, 500)
            final_list, process_list = self.sample_batch(None, to_generate, num_nodes=None, min_n = min_n, max_n = max_n)
            samples.extend(final_list)
            process_samples.extend(process_list)
            samples_num_to_generate -= to_generate

        print("saving graph")
        dict_new = self._load_event_ontology()
        gs = []
        for idx, item in enumerate(samples):
            g = igraph.Graph(directed=True)
           
            n = item[0].shape[0]
            g.add_vertices(n)
            for i, node_type in enumerate(item[0]):
                g.vs[i]['type'] = dict_new[node_type.item()]
            for i, edge_list in enumerate(item[1]):
                for j, edge in enumerate(edge_list):
                    if edge == 1:
                        g.add_edge(i,j)
            g = self._reorder_graph(g)
            gs.append(g)
        with open("../result_" + self.args.data_type + "/dataset_{}_seed_{}".format(self.args.dataset, self.args.seed) + '/predict_igraph.pkl', 'wb') as handle:
            pickle.dump(gs, handle)     
        print("save in " + "../result_" + self.args.data_type + "/dataset_{}_seed_{}".format(self.args.dataset, self.args.seed))
        print("Done.")

        if self.args.save_process_pic:
            print("generate process")
            print(len(process_list))
            for ids, sampl in enumerate(process_samples):
                gs = []
                for idx, item in enumerate(sampl):
                    print(idx)
                    g = igraph.Graph(directed=True)
                
                    n = item[0].shape[0]
                    g.add_vertices(n)
                    for i, node_type in enumerate(item[0]):
                        g.vs[i]['type'] = dict_new[node_type.item()]
                    for i, edge_list in enumerate(item[1]):
                        for j, edge in enumerate(edge_list):
                            if edge == 1:
                                g.add_edge(i,j)
                    g = self._reorder_graph(g)
                    gs.append(g)
                with open("../result_" + self.args.data_type + "/dataset_{}_seed_{}".format(self.args.dataset, self.args.seed) +  '/process_predict_igraph'+str(ids)+'.pkl', 'wb') as handle:
                    pickle.dump([gs,sampl], handle)     
                print("save in " + "../result_" + self.args.data_type + "/dataset_{}_seed_{}".format(self.args.dataset, self.args.seed) +  '/process_predict_igraph'+str(ids)+'.pkl')
                print("Done.")
      
    def _load_event_ontology(self):
        saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
        event_types_ontology = saved_dict['event_types']
        event_types_ontology_new = {0: "START", 1: "END"}
        for key, val in event_types_ontology.items():
            event_types_ontology_new[val + 2] = key
        assert len(event_types_ontology_new) == 69
                
        return event_types_ontology_new

    def _reorder_graph(self, graph):
        new_order = graph.topological_sorting(mode='out')

        new_g = igraph.Graph(directed=True)
        new_g.add_vertices(len(list(graph.vs)))

        new_order_dict = {}
        for ii in range(len(new_order)):
            new_g.vs[ii]['type'] = graph.vs[new_order[ii]]['type']
            new_order_dict[new_order[ii]] = ii
        all_prev_edges = list(graph.es)

        for ii in range(len(all_prev_edges)):
            curr_edge = all_prev_edges[ii]
            new_g.add_edge(new_order_dict[curr_edge.source], new_order_dict[curr_edge.target])

        return new_g

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  
        probE = E @ Qtb.E.unsqueeze(1)  

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)
        
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        
        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data


    def forward(self, noisy_data, node_mask, depth = None):
        X = noisy_data['X_t'].float()
        E = noisy_data['E_t'].float()
        y = torch.hstack((noisy_data['y_t'], torch.ones([noisy_data['y_t'].size(0),1]).cuda() * noisy_data['t'])).float()

        return self.model(X, E, y, node_mask, depth)

    @torch.no_grad()
    def sample_batch(self, type_c, batch_size: int, num_nodes=None, min_n = 15, max_n = 30):
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_nodes = torch.clamp(n_nodes, min=min_n,max=max_n)
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        if type_c == None:
            X, E, y = z_T.X, z_T.E, (torch.zeros([node_mask.size(0), 0])).type_as(node_mask.long())
        else:
            X, E, y = z_T.X, z_T.E, (torch.ones((node_mask.size(0), 1)) * type_c).type_as(node_mask.long())

        if self.args.use_pos:
            depth = self.depth_sampler.sample_n(X.size(0), n_nodes, X.size(1))

        else:
            depth = None
        
        process_list = [[]for i in range(int(self.T/100)+1)]
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        iters = 0
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            if s_int % 100 == 0 or s_int == self.T - 1:
                sample_save = utils.PlaceHolder(X=X, E=E, y=y)
                sample_save = sample_save.mask(node_mask, collapse=True)
                sX, sE, sy = sample_save.X, sample_save.E, sample_save.y
                for i in range(batch_size):
                    n = n_nodes[i]
                    atom_types = sX[i, :n].cpu()
                    edge_types = sE[i, :n, :n].cpu()
                    process_list[iters].append([atom_types, edge_types])
                iters += 1
            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, n_nodes, depth)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        predicted_graph_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            predicted_graph_list.append([atom_types, edge_types])
        return molecule_list,process_list
        # return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, n_nodes, depth):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}

        pred = self.forward(noisy_data, node_mask, depth = depth)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1
        # print(prob_X.shape)
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()


        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_t)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
