import os
import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch


import pickle
import numpy as np
import random
import os



def seed_everything(seed=1234):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def to_dense(x, edge_index, edge_attr, batch):
    """
    The code is copied and adapted from https://arxiv.org/abs/2209.14734
    """
    X, node_mask = to_dense_batch(x=x, batch=batch) # bs, max_n ，f

    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)# bs, max_n, max_n, f

    E = encode_no_edge(E)
    
    return PlaceHolder(X=X, E=E, y=None), node_mask

def encode_no_edge(E):
    """
    The code is copied and adapted from https://arxiv.org/abs/2209.14734
    """
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    assert first_elt.sum() <= 1e-4
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            # print(self.E[0])
            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

def to_nodense(matrix):
    new = torch.zeros(matrix.size(0), matrix.size(1), matrix.size(2), 134)
    # print(new.shape)
    for b in range(len(matrix)):
        for i in range(len(matrix[b])):
            for j in range(len(matrix[b][i])):
                new[b][i][j][int(matrix[b][i][j].item())] = 1
    return new

def load_graphs_with_type_label(args , name, max_n_data = 50):
    # load ENAS format NNs to igraphs or tensors
    g_list_train = []
    max_n = 0  # maximum number of nodes

    
    print("load training graphs:")
    prefix = None
    if args.data_type == "None":
        prefix = "_remove_same_type_and_unlogic"
    elif args.data_type == "IGE":
        prefix = "_remove_same_type_knowledge_inject_p_6"
    elif args.data_type == "json":
        prefix = "_remove_same_type_knowledge_inject_p_6_json"
        
    
    print('../../data/Wiki_IED_split/train/' + 'suicide_ied' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl')
    print('../../data/Wiki_IED_split/train/' + 'wiki_ied_bombings' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl')
    print('../../data/Wiki_IED_split/train/' + 'wiki_mass_car_bombings' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl')
    [graphs_train1, depth1] = pickle.load(
        open('../../data/Wiki_IED_split/train/' + 'suicide_ied' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl', 'rb'))
    [graphs_train2, depth2] = pickle.load(
        open('../../data/Wiki_IED_split/train/' + 'wiki_ied_bombings' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl', 'rb'))
    [graphs_train3, depth3] = pickle.load(
        open('../../data/Wiki_IED_split/train/' + 'wiki_mass_car_bombings' + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl', 'rb'))
    
    graphs_train = [graphs_train1, graphs_train2, graphs_train3]
    depth = [depth1, depth2, depth3]

    
    for mm in range(len(graphs_train)):
        for i, row in enumerate(graphs_train[mm]):
            g, n = row
            max_n = max(max_n, n)
            g_list_train.append([g, depth[mm][i], mm * 1.0])
    print("load testing graphs:")
    print('../../data/Wiki_IED_split/test/' + 'suicide_ied' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl')
    print('../../data/Wiki_IED_split/test/' + 'wiki_ied_bombings' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl')
    print('../../data/Wiki_IED_split/test/' + 'wiki_mass_car_bombings' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl')
    g_list_test = [[], [], []]
    
    [graphs_test1, depth1] = pickle.load(
        open('../../data/Wiki_IED_split/test/' + 'suicide_ied' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl', 'rb'))
    [graphs_test2, depth2] = pickle.load(
        open('../../data/Wiki_IED_split/test/' + 'wiki_ied_bombings' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl', 'rb'))
    [graphs_test3, depth3] = pickle.load(
        open('../../data/Wiki_IED_split/test/' + 'wiki_mass_car_bombings' + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl', 'rb'))
    graphs_test = [graphs_test1, graphs_test2, graphs_test3]
    depth = [depth1, depth2, depth3]
    # assert len(graphs_test) == len(depth)
    
    for mm in range(len(graphs_test)):
        for i, row in enumerate(graphs_test[mm]):
            g, n = row
            max_n = max(max_n, n)
            g_list_test[mm].append([g, depth[mm][i], mm * 1.0])
    # print(len(g_list_test))
    if args.data_type == "r3":
        args.num_vertex_type = args.event_node_type + 2  # original types + start/end types
    else:
        args.num_vertex_type = args.event_node_type + 2  # original types + start/end types
    args.max_n = max_n  # maximum number of nodes
    args.START_TYPE = 0  # predefined start vertex type
    args.END_TYPE = 1 # predefined end vertex type

    print('# node types: %d' % args.num_vertex_type)
    print('maximum # nodes: %d' % args.max_n)

    random.shuffle(g_list_train)
    # random.Random(rand_seed).shuffle(g_list_test)

    return g_list_train, g_list_test

def load_graphs_with_type_label_single(args , name, max_n_data = 50):
    # load ENAS format NNs to igraphs or tensors
    g_list_train = []
    max_n = 0  # maximum number of nodes

    prefix = None
    if args.data_type == "None":
        prefix = "_remove_same_type_and_unlogic"
    elif args.data_type == "IGE":
        prefix = "_remove_same_type_knowledge_inject_p_6"
    elif args.data_type == "json":
        prefix = "_remove_same_type_knowledge_inject_p_6_json"
        
    print("load training graphs:")
    print('../../data/Wiki_IED_split/train/' + name + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl')
    [graphs_train1, depth1] = pickle.load(
        open('../../data/Wiki_IED_split/train/' + name + '_train_pruned_new_no_iso_max_' + str(int(max_n_data)) + prefix + '_igraphs.pkl', 'rb'))
    print(len(graphs_train1))
    graphs_train = [graphs_train1]
    depth = [depth1]

    
    for mm in range(len(graphs_train)):
        for i, row in enumerate(graphs_train[mm]):
            g, n = row
            max_n = max(max_n, n)
            g_list_train.append([g, depth[mm][i], mm * 1.0])
    
    print("load testing graphs:")
    print('../../data/Wiki_IED_split/test/' + name + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl')
    g_list_test = []
    
    [graphs_test1, depth1] = pickle.load(
        open('../../data/Wiki_IED_split/test/' + name + '_test_human_pruned_new_no_iso_max_' + str(int(max_n_data)) +'_igraphs.pkl', 'rb'))

    graphs_test = [graphs_test1]
    depth = [depth1]
    # assert len(graphs_test) == len(depth)
    
    for mm in range(len(graphs_test)):
        for i, row in enumerate(graphs_test[mm]):
            g, n = row
            max_n = max(max_n, n)
            g_list_test.append([g, depth[mm][i], mm * 1.0])


    args.num_vertex_type = args.event_node_type + 2  # original types + start/end types
    args.max_n = max_n  # maximum number of nodes
    args.START_TYPE = 0  # predefined start vertex type
    args.END_TYPE = 1 # predefined end vertex type

    print('# node types: %d' % args.num_vertex_type)
    print('maximum # nodes: %d' % args.max_n)

    random.shuffle(g_list_train)
    # random.Random(rand_seed).shuffle(g_list_test)

    return g_list_train, g_list_test

def load_graphs_with_emb():

    graphs_train = pickle.load(
        open('../../data/train_pruned_with_bert_max_50_set_enriched_wo_event.pkl', 'rb'))
    graphs_test = pickle.load(
        open('../../data/test_pruned_with_bert_max_50_set_enriched_wo_event.pkl', 'rb'))
    return graphs_train, graphs_test

class depth_sampler:
    def __init__(self, depth):
        self.depth = depth
        self.depth = depth / depth.sum()
        self.m = torch.distributions.Categorical(depth)

    def sample_n(self, b, n, l):
        d_sample = torch.zeros(b,l)
        for i in range(b):
            for j in range(n[i]):
                idx = self.m.sample()
                d_sample[i][j]=idx
            d_sample[i][:n[i]], _ = d_sample[i][:n[i]].sort()
            d_sample[i][0] = 0
        return d_sample