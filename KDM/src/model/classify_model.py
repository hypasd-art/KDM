import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils
from model.transformer_classify_model import c_GraphTransformer


class CrossEntropyMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_ce=torch.tensor(0.)
        self.total_samples=torch.tensor(0.)

    def forward(self, preds, target):
        target = torch.argmax(target, dim=-1)
        type_set = set(target.cpu().numpy())
        output = 0
        for type_id in type_set:
            edge_mask = (target == type_id)
            weight = edge_mask.int().sum() / target.size(0)
            weight = weight if weight < 0.9 else 0.9
            weight = weight if weight > 0.1 else 0.1
            output += F.cross_entropy(preds[edge_mask,:], target[edge_mask], reduction='sum')/weight
       
        res = output/preds.size(0)
        self.total_ce += output.cpu()
        self.total_samples += preds.size(0)
        return res

    def compute(self):
        return self.total_ce / self.total_samples
    
    def reset(self):
        self.total_ce = torch.tensor(0.)
        self.total_samples = torch.tensor(0.)

class c_model(nn.Module):
    def __init__(self, args, input_dims, output_dims):
        super().__init__()

        input_dims = input_dims
        output_dims = output_dims

        self.args = args
        self.model_dtype = torch.float32

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.device = torch.device("cuda")
        self.train_loss = CrossEntropyMetric()
        self.test_loss = CrossEntropyMetric()

        self.model = c_GraphTransformer(n_layers=args.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=args.hidden_mlp_dims,
                                      hidden_dims=args.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())
        self.linear_1 = nn.Linear(output_dims['X'], int(output_dims['X']/2))
        self.ac = nn.GELU()
        self.linear_2 = nn.Linear(int(output_dims['X']/2), int(output_dims['X']/2))
        self.dropout = nn.Dropout(0.2)
        self.layernorm = nn.LayerNorm(output_dims['X'])
        self.classifer = nn.Sequential(nn.Linear(output_dims['X'], args.hidden_mlp_dims['X']), nn.ReLU(),
                                      nn.Linear(args.hidden_mlp_dims['X'], output_dims['E']), nn.ReLU())

    def set_gpu(self, data=None, node_mask=None):
        data.X = data.X.cuda()
        data.E = data.E.cuda()
        node_mask = node_mask.cuda()
        return data, node_mask
    
    def loss(self, data, i):
        len_event = data.len_event
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        assert (False not in node_mask)

        dense_data, node_mask = self.set_gpu(dense_data, node_mask)
        dense_data = dense_data.mask(node_mask)

        features, adj_init, true_E = dense_data.X, dense_data.E, data.adj_true
        assert len_event < true_E.size(0)

        true_E = F.one_hot(true_E.to(torch.int64)).cuda()
        dis = data.dis.unsqueeze(0).unsqueeze(-1)
        pred = self.model(features, adj_init, node_mask, dis)
        
        masked_pred_E = pred.E

        if self.args.use_classifer:
            pred_X = pred.X

            role = data.role.unsqueeze(0).unsqueeze(-1).expand(-1, -1, pred_X.size(-1)).clone()
            div_term = (torch.arange(0, pred_X.size(-1), 2).float() * -(math.log(10000.0) / pred_X.size(-1))).exp()
            
            role[:,:,0::2] = torch.sin(role[:, :, 0::2] * div_term[:role[:, :, 0::2].size(-1)])
            role[:, :, 1::2] = torch.cos(role[:, :, 1::2] * div_term[:role[:, :, 1::2].size(-1)])
            assert role.requires_grad == False
            pred_X = pred_X + role.cuda()
            Pred_X = self.layernorm(pred_X)
            pred_X = self.linear_1(pred_X)
            pred_X = self.ac(pred_X)
            pred_X = self.dropout(pred_X)
            pred_X = self.linear_2(pred_X)
            X1 = pred_X.unsqueeze(2)  # b, n, 1 , f
            X1 = X1.expand(-1,-1, X1.size(1), -1) # b, n, n, f
            X2 = pred_X.unsqueeze(1)  # b, 1, n, f
            X2 = X2.expand(-1, X2.size(2), -1, -1)
            
            masked_pred_E = torch.cat([X1, X2], dim=-1) #+ dis.cuda()
            
            masked_pred_E = self.classifer(masked_pred_E)
            

        event_mask = torch.zeros(true_E.size(0), true_E.size(1))
        event_mask[len_event:, len_event:]= 1

        true_E = true_E * event_mask.unsqueeze(-1).cuda()
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        # Remove masked rows
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        loss_E = self.train_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        return loss_E

    def test(self, data):
        len_event = data.len_event
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        assert (False not in node_mask)

        dense_data, node_mask = self.set_gpu(dense_data, node_mask)
        dense_data = dense_data.mask(node_mask)

        features, adj_init, true_E = dense_data.X, dense_data.E, data.adj_true
        assert len_event < true_E.size(0)
        true_E = F.one_hot(true_E.to(torch.int64)).cuda()

        dis = data.dis.unsqueeze(0).unsqueeze(-1)

        pred = self.model(features, adj_init, node_mask, dis)
        masked_pred_E = pred.E
        if self.args.use_classifer:
            pred_X = pred.X
            role = data.role.unsqueeze(0).unsqueeze(-1).expand(-1, -1, pred_X.size(-1)).clone()
            div_term = (torch.arange(0, pred_X.size(-1), 2).float() * -(math.log(10000.0) / pred_X.size(-1))).exp()
            
            role[:,:,0::2] = torch.sin(role[:, :, 0::2] * div_term[:role[:, :, 0::2].size(-1)])
            role[:, :, 1::2] = torch.cos(role[:, :, 1::2] * div_term[:role[:, :, 1::2].size(-1)])
            assert role.requires_grad == False
            pred_X = pred_X + role.cuda()

            
            Pred_X = self.layernorm(pred_X)
            pred_X = self.linear_1(pred_X)
            pred_X = self.ac(pred_X)
            pred_X = self.dropout(pred_X)
            pred_X = self.linear_2(pred_X)
            X1 = pred_X.unsqueeze(2)  # b, n, 1 , f
            X1 = X1.expand(-1,-1, X1.size(1), -1) # b, n, n, f
            X2 = pred_X.unsqueeze(1)  # b, 1, n, f
            X2 = X2.expand(-1, X2.size(2), -1, -1)
            
            masked_pred_E = torch.cat([X1, X2], dim=-1)
            
            masked_pred_E = self.classifer(masked_pred_E)

        event_mask = torch.zeros(true_E.size(0), true_E.size(1))
        event_mask[len_event:, len_event:]= 1

        true_E = true_E * event_mask.unsqueeze(-1).cuda()
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        loss_E = self.test_loss(flat_pred_E, flat_true_E) if true_E.numel() > 0 else 0.0
        flat_pred_E = torch.argmax(flat_pred_E, dim=-1).flatten().cpu()
        flat_true_E = torch.argmax(flat_true_E, dim=-1).flatten().cpu()
        return loss_E, flat_pred_E, flat_true_E
        

    
    def predict(self, data):
        len_event = data.len_event
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        assert (False not in node_mask)

        dense_data, node_mask = self.set_gpu(dense_data, node_mask)
        dense_data = dense_data.mask(node_mask)

        features, adj_init = dense_data.X, dense_data.E
        dis = data.dis.unsqueeze(0).unsqueeze(-1)
        dis_b_mask = ((data.dis != 4)).unsqueeze(0)
        
        pred = self.model(features, adj_init, node_mask, dis)
        masked_pred_E = pred.E

        if self.args.use_classifer:
            pred_X = pred.X
            role = data.role.unsqueeze(0).unsqueeze(-1).expand(-1, -1, pred_X.size(-1)).clone()
            div_term = (torch.arange(0, pred_X.size(-1), 2).float() * -(math.log(10000.0) / pred_X.size(-1))).exp()
            
            role[:,:,0::2] = torch.sin(role[:, :, 0::2] * div_term[:role[:, :, 0::2].size(-1)])
            role[:, :, 1::2] = torch.cos(role[:, :, 1::2] * div_term[:role[:, :, 1::2].size(-1)])
            assert role.requires_grad == False
            pred_X = pred_X + role.cuda()

            Pred_X = self.layernorm(pred_X)
            pred_X = self.linear_1(pred_X)
            pred_X = self.ac(pred_X)
            pred_X = self.dropout(pred_X)
            pred_X = self.linear_2(pred_X)
            X1 = pred_X.unsqueeze(2)  # b, n, 1 , f
            X1 = X1.expand(-1,-1, X1.size(1), -1) # b, n, n, f
            X2 = pred_X.unsqueeze(1)  # b, 1, n, f
            X2 = X2.expand(-1, X2.size(2), -1, -1)
            
            masked_pred_E = torch.cat([X1, X2], dim=-1)# + dis.cuda()
            
            masked_pred_E = self.classifer(masked_pred_E)
        masked_pred_E[dis_b_mask] = 0
        for ii in range(dis_b_mask.size(0)):
            for jj in range(dis_b_mask.size(1)):
                for kk in range(dis_b_mask.size(2)):
                    if dis_b_mask[ii,jj,kk]:
                        masked_pred_E[ii,jj,kk,1] = 1

        adj_init[:,len_event:, len_event:,:]= masked_pred_E[:,len_event:, len_event:,:]

        return adj_init
