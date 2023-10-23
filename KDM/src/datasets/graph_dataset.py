import torch
from torch.utils.data import Dataset
import torch_geometric.utils



class GraphDataset(Dataset):
    def __init__(self, args, data_list):
        
        self.args = args
        self.data_list = data_list
        self.data_processed = []
        self.adjs = []
        self.max_depth = 0
        self.process()
        print(f'*****Dataset created*****')

    def process(self):
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            depth = torch.Tensor(graph[1])
            if self.max_depth < max(depth):
                self.max_depth = max(depth)
            label = graph[2]

            pad_zero = torch.zeros(self.args.max_n - len(depth))
            depth = torch.cat([depth, pad_zero], dim = -1)
            
            X = torch.zeros(len(g.vs), self.args.num_vertex_type)
            adj = torch.zeros(len(g.vs), len(g.vs), dtype=torch.float)
            y = torch.zeros([1, 1]).float()
            y[0][0] = label
            for edge_tuple in g.get_edgelist():
                adj[edge_tuple[0], edge_tuple[1]] = 1.0
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], self.args.num_edge_type, dtype=torch.float)
            for i in edge_attr:
                i[1] = 1.0
            
            for i, node in enumerate(g.vs):
                X[i][node['type']] = 1
            assert X[0][0] == 1
            assert X[len(g.vs)-1][1] == 1
            
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, depth=depth, idx=i)
            self.data_processed.append(data)

    def __len__(self):
        return len(self.data_processed)

    def __getitem__(self, idx):
        return self.data_processed[idx]

    def calcu_info_depth(self):
        info_depth = [0 for i in range(1 + int(self.max_depth.item()))]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            depth = graph[1]
            for k in depth:
                info_depth[k] += 1
        return torch.Tensor(info_depth)
    
    def calcu_info_type(self):
        info_type = [0 for i in range(self.args.num_vertex_type)]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            for node in g.vs:
                info_type[node['type']]+=1
          
        return torch.Tensor(info_type)

    def calcu_info_num(self):
        info_num = [0 for i in range(1 + self.args.max_n)]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            info_num[len(g.vs)] +=1
        return torch.Tensor(info_num)

class GraphDataset_single(Dataset):
    def __init__(self, args, data_list):
        self.args = args
        self.data_list = data_list
        self.data_processed = []
        self.adjs = []
        self.max_depth = 0
        self.process()
        print(f'*****Dataset created*****')

    def process(self):
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            depth = torch.Tensor(graph[1])
            if self.max_depth < max(depth):
                self.max_depth = max(depth)
            label = graph[2]
            assert label == 0

            pad_zero = torch.zeros(self.args.max_n - len(depth))
            depth = torch.cat([depth, pad_zero], dim = -1)
            
            X = torch.zeros(len(g.vs), self.args.num_vertex_type)
            adj = torch.zeros(len(g.vs), len(g.vs), dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            for edge_tuple in g.get_edgelist():
                adj[edge_tuple[0], edge_tuple[1]] = 1.0
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], self.args.num_edge_type, dtype=torch.float)
            for i in edge_attr:
                i[1] = 1.0
            
            for i, node in enumerate(g.vs):
                X[i][node['type']] = 1
            assert X[0][0] == 1
            assert X[len(g.vs)-1][1] == 1
            assert depth[0] == 0
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, depth=depth, idx=i)
            self.data_processed.append(data)

    def __len__(self):
        return len(self.data_processed)

    def __getitem__(self, idx):
        return self.data_processed[idx]

    def calcu_info_depth(self):
        info_depth = [0 for i in range(1 + int(self.max_depth.item()))]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            depth = graph[1]
            for k in depth:
                info_depth[k] += 1
        return torch.Tensor(info_depth)
    
    def calcu_info_type(self):
        info_type = [0 for i in range(self.args.num_vertex_type)]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            for node in g.vs:
                info_type[node['type']]+=1
          
        return torch.Tensor(info_type)

    def calcu_info_num(self):
        info_num = [0 for i in range(1 + self.args.max_n)]
        for i, graph in enumerate(self.data_list):
            g = graph[0]
            info_num[len(g.vs)] +=1
        return torch.Tensor(info_num)

class c_GraphDataset(Dataset):
    def __init__(self, args, features, adj_true, adj_init, len_event, dis, role):
        
        self.args = args
        self.features = features
        self.adj_true = adj_true
        self.adj_init = adj_init
        self.len_event = len_event
        self.dis = dis
        self.role = role
        assert len(self.len_event) == len(self.features)
        assert len(self.role) == len(self.len_event)
      
        self.data_processed = []
        self.process()
        print(f'*****Dataset created*****')

    def process(self):
        for i in range(len(self.features)):
            X = torch.Tensor(self.features[i])
            if X.size(0) > 400 or X.size(0) == 0:
                continue
            edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(torch.Tensor(self.adj_init[i]))
            edge_attr2 = torch.zeros(edge_index.shape[-1], 134, dtype=torch.float)
            for j, v in enumerate(edge_attr2):
                v[int(edge_attr[j])] = 1.0
            
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, dis=torch.Tensor(self.dis[i]), role=torch.Tensor(self.role[i]),
                                                                                            edge_attr=edge_attr2, len_event = torch.tensor(self.len_event[i]), 
                                                                                            adj_true=torch.Tensor(self.adj_true[i]), idx=i)
            self.data_processed.append(data)

    def __len__(self):
        return len(self.data_processed)

    def __getitem__(self, idx):
        return self.data_processed[idx]

class c_predict_GraphDataset(Dataset):
    def __init__(self, args, features, adj_init, len_event, dis, role):
        
        self.args = args
        self.features = features
        self.adj_init = adj_init
        self.len_event = len_event
        self.dis = dis
        self.role = role
        assert len(self.len_event) == len(self.features)
        assert len(self.role) == len(self.len_event)
      
        self.data_processed = []
        self.process()
        print(f'*****Dataset created*****')

    def process(self):
        for i in range(len(self.features)):
            X = torch.Tensor(self.features[i])
            edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(torch.Tensor(self.adj_init[i]))
            edge_attr2 = torch.zeros(edge_index.shape[-1], 134, dtype=torch.float)
            for j, v in enumerate(edge_attr2):
                v[int(edge_attr[j])] = 1.0
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr2, len_event = torch.tensor(self.len_event[i]), 
                                                        dis=torch.Tensor(self.dis[i]), role=torch.Tensor(self.role[i]),idx=i)
            self.data_processed.append(data)

    def __len__(self):
        return len(self.data_processed)

    def __getitem__(self, idx):
        return self.data_processed[idx]
