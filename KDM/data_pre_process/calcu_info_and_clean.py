import json
import numpy as np
import pickle
import pandas as pd
import igraph
import networkx as nx
import matplotlib.pyplot as plt
from igraph import *
max_node = 50
max_d_m = 10
print("max_node_num = " + str(max_node))

remove_edge_list = [('Life.Die.Unspecified', 'Life.Injure.Unspecified'),
                    ('Justice.ArrestJailDetain.Unspecified', 'Conflict.Attack.DetonateExplode'),
                    ('Justice.ArrestJailDetain.Unspecified', 'Conflict.Attack.Unspecified'),
                    ('Personnel.EndPosition.Unspecified', 'Personnel.StartPosition.Unspecified'),
                    ('Justice.Sentence.Unspecified', 'Life.Die.Unspecified'),
                    ('Contact.ThreatenCoerce.Unspecified', 'Justice.ReleaseParole.Unspecified')]


def get_seq(res, dict_event):
    res_seq_set2 = []
    res_seq_set3 = []
    res_set2 = {}
    res_set3 = {}
    for i in res.get_edgelist():
        if res.vs[i[0]]['type'] == 0 or res.vs[i[1]]['type'] == 1:
            continue
        res_seq_set2.append(i)
   
    for e in list(res_seq_set2):
        for i in list(res_seq_set2):
            if e[1] == i[0]:
                seq = (e[0],e[1],i[1])
                res_seq_set3.append(seq)
    for e in res_seq_set2:
        r = (dict_event[res.vs[e[0]]['type']], dict_event[res.vs[e[1]]['type']])
        if r in res_set2:
            res_set2[r] += 1
        else:
            res_set2[r] = 1
    for e in res_seq_set3:
        r = (dict_event[res.vs[e[0]]['type']], dict_event[res.vs[e[1]]['type']], dict_event[res.vs[e[2]]['type']])
        if r in res_set3:
            res_set3[r] += 1
        else:
            res_set3[r] = 1
    return res_set2, res_set3

def get_node_type(res, dict_event):
    node_dict = {}
    for i, node in enumerate(res.vs):
        if dict_event[node['type']] not in node_dict:
            node_dict[dict_event[node['type']]] = 1
        else:
            node_dict[dict_event[node['type']]] += 1
    return node_dict 

def get_all_dict_info(g_list):
    event_types_ontology_new = _load_event_ontology()
    dict_seq2 = {}
    dict_seq3 = {}
    node_q = {}
    new_g_list = []
    for [graph, depth] in g_list:
        new_graph = remove_same_type_edge(graph, event_types_ontology_new)
        res_dict2, res_dict3 = get_seq(new_graph, event_types_ontology_new)
        node_dict = get_node_type(new_graph, event_types_ontology_new)
        for k, v in node_dict.items():
            if k in node_q:
                node_q[k] += v
            else:
                node_q[k] = v
        if len(res_dict2) == 0:
            continue
        for k, v in res_dict2.items():
            if k in dict_seq2:
                dict_seq2[k] += v
            else:
                dict_seq2[k] = v
        for k, v in res_dict3.items():
            if k in dict_seq3:
                dict_seq3[k] += v
            else:
                dict_seq3[k] = v
        if len(new_graph.vs) <= 2:
            continue
        new_g_list.append((new_graph, len(list(new_graph.vs))))
    depth = get_node_depth(new_g_list, event_types_ontology_new)
    dict_seq2 = sorted(dict_seq2.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    dict_seq3 = sorted(dict_seq3.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    node_q = sorted(node_q.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    print(node_q)
    return new_g_list, depth, dict_seq2, dict_seq3

def _load_event_ontology():
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {0: "START", 1: "END"}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[val + 2] = key
    return event_types_ontology_new

def remove_same_type_edge(graph, dict_event):

    judge = True
    while judge:
        judge = False
        for edge in graph.get_edgelist():
            if graph.vs[edge[0]]['type'] == graph.vs[edge[1]]['type']:
                judge = True
                graph.delete_edges((edge[0], edge[1]))
                break

    judge = True
    while judge:
        judge = False
        for edge in graph.get_edgelist():
            if (dict_event[graph.vs[edge[0]]['type']], dict_event[graph.vs[edge[1]]['type']]) in remove_edge_list:
                judge = True
                graph.delete_edges((edge[0], edge[1]))
                break  
    
    # 删除孤立点
    graph = delete_iso_node(graph)
    # 检查连通性

    for edge in graph.get_edgelist():
        if graph.vs[edge[0]]['type'] == graph.vs[edge[1]]['type']:
            print("yep")
    return graph

def delete_iso_node(graph):
    while True:
        judge = False
        for i in range(len(graph.vs)):
            node = graph.vs[i]
            if node['type'] == 0 or node['type'] == 1:
                graph.delete_vertices(node)
                judge = True
                break
        if judge == False:
            break
    
    indegree_0, outdegree_0 = _get_nodes_for_start_end(graph, len(graph.vs))
    # 删掉独立点
    degree_0 = indegree_0.intersection(outdegree_0)
    # print(degree_0)
    num_before = len(graph.vs)
    # print(degree_0)
    graph.delete_vertices(list(degree_0))
    assert num_before - len(degree_0) == len(graph.vs)

    indegree_0, outdegree_0 = _get_nodes_for_start_end(graph, len(graph.vs))
    num = len(graph.vs)
    graph.add_vertices(2)
    graph.vs[num]['type'] = 0
    graph.vs[num+1]["type"] = 1

    for ii in range(0, num):
        if ii in indegree_0:
            # print(ii)
            graph.add_edge(num, ii)
        if ii in outdegree_0:
            graph.add_edge(ii, num+1)
    graph = _reorder_graph(graph)
    return graph

def _get_nodes_for_start_end(graph, node_num):
    indegree_0 = []
    outdegree_0 = []
    for ii in range(0, node_num):
        if graph.vs[ii].indegree() == 0:
            indegree_0.append(ii)
        if graph.vs[ii].outdegree() == 0:
            outdegree_0.append(ii)
    return set(indegree_0), set(outdegree_0)

def _reorder_graph(graph):

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

def get_node_depth(graphs, dict_event):
    depth_cal = [[{},{}] for i in range(len(dict_event))]
    path_depth = []
    max_d = 0
    for i, row in enumerate(graphs):
        g, n = row
        start = None
        end = None
        graph_depth = []
        for i, node in enumerate(g.vs):
            if node['type'] == 0:
                start = i
            if node['type'] == 1:
                end = i 
        assert start == 0
        
        assert end == len(g.vs)-1
        for i, node in enumerate(g.vs):
            depth = 0
            path = g.get_all_simple_paths(start, to=i)
            path_end = g.get_all_simple_paths(i, to=end)
            d = int(mean([len(path[i]) for i in range(len(path))]))
            if d > max_d:
                max_d = d 
            graph_depth.append(d)
        path_depth.append(graph_depth)
    print(f"max depth {max_d}")
    return path_depth

def construct_event_igraphs_by_remove_same_edge(name, dataset, max_n_data = 50):
    print(name, dataset)
    g_list_train = []
    [graphs_train, depth] = pickle.load(
        open('../../data/Wiki_IED_split/' + dataset +'/' + name + '_' + dataset +'_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_igraphs.pkl', 'rb'))
    assert len(graphs_train) == len(depth)
    graphs_train = [graphs_train]
    
    for mm in range(len(graphs_train)):
        for i, row in enumerate(graphs_train[mm]):
            g, n = row
            g_list_train.append([g, depth[i]])
    new_g_list, depth, _, _ = get_all_dict_info(g_list_train)

    with open('../../data/Wiki_IED_split/' + dataset +'/'+ name + '_' + dataset +'_pruned_new_no_iso_max_'+ str(int(max_n_data)) + '_remove_same_type_and_unlogic_igraphs.pkl', 'wb') as handle:
        pickle.dump([new_g_list, depth], handle)
    

if __name__ == "__main__":
    file_dir = "../../data/Wiki_IED_split/"
    dataset_types = ["test", "dev", "train"]
    file_names = ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]
    
    for file_name in file_names:
        for dataset in dataset_types:
            construct_event_igraphs_by_remove_same_edge(file_name, dataset, max_node)

