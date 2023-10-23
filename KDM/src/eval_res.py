"""
Part of code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import numpy as np
import pickle
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def _load_event_ontology():
        saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
        event_types_ontology = saved_dict['event_types']
        event_types_ontology_new = {"START": 0, "END": 1}
        for key, val in event_types_ontology.items():
            event_types_ontology_new[key] = val + 2
        return event_types_ontology_new

def calcu_event_match(res, ans):
    """
    :param res: igraph
    :param ans: igraph
    :return: int
    """
    res_set = set()
    ans_set = set()
    for node in res.vs:
        res_set.add(node['type'])
    for node in ans.vs:
        ans_set.add(node['type'])
    p = len(res_set.intersection(ans_set))/len(ans_set)
    r = len(res_set.intersection(ans_set))/len(res_set)
    if p + r == 0:
        return 0, 0, 0
    f = 2 * p * r/ (p + r)
    return p,r,f

def calcu_seq_match(res, ans):
    res_seq_set2 = set()
    ans_seq_set2 = set()
    res_seq_set3 = set()
    ans_seq_set3 = set()
    res_set2 = set()
    ans_set2 = set()
    res_set3 = set()
    ans_set3 = set()
    for seq in res.get_edgelist():
        res_seq_set2.add(seq)
    for seq in ans.get_edgelist():
        ans_seq_set2.add(seq)
    for e in list(res_seq_set2):
        for i in list(res_seq_set2):
            if e[1] == i[0]:
                res_seq_set3.add((e[0],e[1],i[1]))
    for e in list(ans_seq_set2):
        for i in list(ans_seq_set2):
            if e[1] == i[0]:
                ans_seq_set3.add((e[0],e[1],i[1]))
    for e in res_seq_set2:
        res_set2.add((res.vs[e[0]]['type'], res.vs[e[1]]['type']))
    for e in res_seq_set3:
        res_set3.add((res.vs[e[0]]['type'], res.vs[e[1]]['type'], res.vs[e[2]]['type']))
    for e in ans_seq_set2:
        ans_set2.add((ans.vs[e[0]]['type'], ans.vs[e[1]]['type']))
    for e in ans_seq_set3:
        ans_set3.add((ans.vs[e[0]]['type'], ans.vs[e[1]]['type'], ans.vs[e[2]]['type']))
    p2 = len(res_set2.intersection(ans_set2))/len(ans_set2) if len(ans_set2) > 0 else 0
    r2 = len(res_set2.intersection(ans_set2))/len(res_set2) if len(res_set2) > 0 else 0
    if p2 + r2 == 0:
        return 0, 0
    f2 = 2 * p2 * r2 / (p2 + r2)
    if len(res_set3) ==0:
        return f2, 0
    p3 = len(res_set3.intersection(ans_set3))/len(ans_set3)
    r3 = len(res_set3.intersection(ans_set3))/len(res_set3)
    if (p3 + r3) == 0:
        return f2, 0
    f3 = 2 * p3 * r3 / (p3 + r3)
    return f2, f3

def collect_node_edge_freq(graphs):
    all_graph_node_freq = []
    all_graph_edge_freq = []
    all_graph_path_3_freq = []
    for ii in range(len(graphs)):
        curr_graph = graphs[ii]
        graph_node_freq = {}
        graph_edge_freq = {}
        node_type_dict = {}
        for idx, node in enumerate(curr_graph.vs):
            curr_type = node["type"]
            node_type_dict[idx] = curr_type
            if curr_type in graph_node_freq:
                graph_node_freq[curr_type] += 1
            else:
                graph_node_freq[curr_type] = 1
        for edge in curr_graph.get_edgelist():
            start_type = node_type_dict[edge[0]]
            end_type = node_type_dict[edge[1]]
            if start_type == end_type:
                continue
            if (start_type, end_type) in graph_edge_freq:
                graph_edge_freq[(start_type, end_type)] += 1
            else:
                graph_edge_freq[(start_type, end_type)] = 1
        
        graph_path_3_freq = {}
        for node in curr_graph.vs:
            curr_node_successors = curr_graph.successors(node)
            curr_type = node["type"]
            for node_n in curr_node_successors:
                curr_node_n_successors = curr_graph.successors(curr_graph.vs[node_n])
                curr_n_type = curr_graph.vs[node_n]["type"]
                for node_n_n in curr_node_n_successors:
                    curr_n_n_type = curr_graph.vs[node_n_n]["type"]
                    if (curr_type, curr_n_type, curr_n_n_type) in graph_path_3_freq:
                        graph_path_3_freq[(curr_type, curr_n_type, curr_n_n_type)] += 1
                    else:
                        graph_path_3_freq[(curr_type, curr_n_type, curr_n_n_type)] = 1
        
        all_graph_node_freq.append(graph_node_freq)
        all_graph_edge_freq.append(graph_edge_freq)
        all_graph_path_3_freq.append(graph_path_3_freq)
    return all_graph_node_freq, all_graph_edge_freq, all_graph_path_3_freq

def compute_node_edge_match(freq_true, freq_pred):
    curr_list_true = list(freq_true.keys())
    curr_list_pred = list(freq_pred.keys())
    intersection = list((Counter(curr_list_true) & Counter(curr_list_pred)).elements())
    if len(curr_list_pred) == 0 or len(curr_list_true) == 0:
        return 0., 0., 0.
    precision = len(intersection) * 1.0 / len(curr_list_pred)
    recall = len(intersection) * 1.0 / len(curr_list_true)
    if precision + recall == 0:
        f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4), round(precision, 4), round(recall, 4)

def node_and_edge_match(curr_scenario_pred_graphs, curr_scenario_true_graphs):
    curr_scenario_pred_node_freq, curr_scenario_pred_edge_freq, curr_scenario_pred_path_3_freq = collect_node_edge_freq(curr_scenario_pred_graphs)
    curr_scenario_true_node_freq, curr_scenario_true_edge_freq, curr_scenario_true_path_3_freq = collect_node_edge_freq(curr_scenario_true_graphs)
    node_f1 = 0.
    node_precision = 0.
    node_recall = 0.
    for mm in range(len(curr_scenario_pred_node_freq)):
        for nn in range(len(curr_scenario_true_node_freq)):
            curr_f1, curr_precision, curr_recall = compute_node_edge_match(curr_scenario_pred_node_freq[mm],curr_scenario_true_node_freq[nn])
            node_f1 += curr_f1
            node_precision += curr_precision
            node_recall += curr_recall
    node_f1 /= len(curr_scenario_pred_node_freq) * len(curr_scenario_true_node_freq)
    node_precision /= len(curr_scenario_pred_node_freq) * len(curr_scenario_true_node_freq)
    node_recall /= len(curr_scenario_pred_node_freq) * len(curr_scenario_true_node_freq)
        
    edge_f1 = 0.
    edge_precision = 0.
    edge_recall = 0.
    for mm in range(len(curr_scenario_pred_edge_freq)):
        for nn in range(len(curr_scenario_true_edge_freq)):
            curr_f1, curr_precision, curr_recall = compute_node_edge_match(curr_scenario_pred_edge_freq[mm], curr_scenario_true_edge_freq[nn])
            edge_f1 += curr_f1
            edge_precision += curr_precision
            edge_recall += curr_recall
    edge_f1 /= len(curr_scenario_pred_edge_freq) * len(curr_scenario_true_edge_freq)
    edge_precision /= len(curr_scenario_pred_edge_freq) * len(curr_scenario_true_edge_freq)
    edge_recall /= len(curr_scenario_pred_edge_freq) * len(curr_scenario_true_edge_freq)
    
    path_3_f1 = 0.
    path_3_precision = 0.
    path_3_recall = 0.
    for mm in range(len(curr_scenario_pred_path_3_freq)):
        for nn in range(len(curr_scenario_true_path_3_freq)):
            curr_f1, curr_precision, curr_recall = compute_node_edge_match(curr_scenario_pred_path_3_freq[mm],curr_scenario_true_path_3_freq[nn])
            path_3_f1 += curr_f1
            path_3_precision += curr_precision
            path_3_recall += curr_recall
    path_3_f1 /= len(curr_scenario_pred_path_3_freq) * len(curr_scenario_true_path_3_freq)
    path_3_precision /= len(curr_scenario_pred_path_3_freq) * len(curr_scenario_true_path_3_freq)
    path_3_recall /= len(curr_scenario_pred_path_3_freq) * len(curr_scenario_true_path_3_freq)

    print("event type match")
    print(round(node_f1, 4), round(node_precision, 4), round(node_recall, 4))
    print("event seq match l = 2")
    print(round(edge_f1, 4), round(edge_precision, 4), round(edge_recall, 4))
    print("event seq match l = 3")
    print(round(path_3_f1, 4), round(path_3_precision, 4), round(path_3_recall, 4))
    return node_f1, edge_f1, path_3_f1

def get_event_and_edge_types(graph, event_types_ontology):
    all_edge_types = {}
    for e in graph.get_edgelist():
        edge_type = (graph.vs[e[0]]["type"], graph.vs[e[1]]["type"])
        if edge_type not in all_edge_types:
            all_edge_types[edge_type] = 1
        else:
            all_edge_types[edge_type] += 1

    all_node_types = {}
    for node in graph.vs:
        node_type = node["type"]
        if node_type not in all_node_types:
            all_node_types[node_type] = 1
        else:
            all_node_types[node_type] += 1
    
    node_distribution = np.zeros(len(event_types_ontology))
    for key, val in all_node_types.items():
        node_distribution[key] = val * 1. / len(graph.vs)
    edge_distribution = np.zeros((len(event_types_ontology), len(event_types_ontology)))
    for key, val in all_edge_types.items():
        edge_distribution[key[0], key[1]] = val / len(graph.es)
    return node_distribution, edge_distribution

def kl_and_MCS(curr_pred_graphs, curr_test_graphs):
    event_types_ontology_new = _load_event_ontology()

    curr_edge_kls = []
    curr_node_kls = []
    for jj in range(len(curr_pred_graphs)):
        for kk in range(len(curr_test_graphs)):
            curr_pred_graph = curr_pred_graphs[jj]
            curr_test_graph = curr_test_graphs[kk]
           
            pred_node_distribution, pred_edge_distribution = get_event_and_edge_types(curr_pred_graph, event_types_ontology_new)
            test_node_distribution, test_edge_distribution = get_event_and_edge_types(curr_test_graph, event_types_ontology_new)
            edge_kl = np.sum(pred_edge_distribution * np.log(
                (pred_edge_distribution + 0.00001) / (test_edge_distribution + 0.00001)))
            node_kl = np.sum(pred_node_distribution * np.log(
                (pred_node_distribution + 0.00001) / (test_node_distribution + 0.00001)))


            curr_edge_kls.append(edge_kl)
            curr_node_kls.append(node_kl)

    print("Node KL:")
    print(round(np.mean(curr_node_kls), 4))
    print("Edge KL:")
    print(round(np.mean(curr_edge_kls), 4))

def evaluation_generation_res_on_origin_test_dataset(file_name, args, type_c = None, seed = 6666):
    print("*"*20 + file_name + "*"*20)
    res_dir = "../result_"+args.data_type+"/dataset_{}_seed_{}".format(file_name, seed) + "/predict_igraph.pkl"
    g_list_test = []
    
    event_types_ontology_new = _load_event_ontology()
    prefix = "_remove_same_type_and_unlogic"
    print("../../data/Wiki_IED_split/test/" + file_name + '_test_pruned_new_no_iso_max_50' + prefix + "_igraphs.pkl")
    # 读取测试数据
    [graphs_test ,depth] = pickle.load(
        open("../../data/Wiki_IED_split/test/" + file_name + '_test_pruned_new_no_iso_max_50' + prefix + "_igraphs.pkl", 'rb'))
   
    graphs_test = [graphs_test]
    for mm in range(len(graphs_test)):
        for i, row in enumerate(graphs_test[mm]):
            g, n = row   
            g_list_test.append(g)
    graph_predict = pickle.load(open(res_dir, 'rb'))
    print(len(graph_predict))
    
    if type_c != None:   
        for idx, g in enumerate(graph_predict):
            nx_graph = nx.DiGraph()
            node_id_to_type = {}
            all_edges = list(g.es)
            for ii in range(len(all_edges)):
                node_id_to_type[all_edges[ii].source] = g.vs[all_edges[ii].source]['type'].split(".")[1] if g.vs[all_edges[ii].source]['type'].split(".")[-1] == 'Unspecified' else g.vs[all_edges[ii].source]['type'].split(".")[-1]
                node_id_to_type[all_edges[ii].target] = g.vs[all_edges[ii].target]['type'].split(".")[1] if g.vs[all_edges[ii].target]['type'].split(".")[-1] == 'Unspecified' else g.vs[all_edges[ii].target]['type'].split(".")[-1]
                nx_graph.add_edge(all_edges[ii].source, all_edges[ii].target)
                nx.draw_networkx(
                    nx_graph, with_labels=True, labels=node_id_to_type, font_size=3,
                    node_size=50, arrowsize=3, width=0.5, pos=nx.spring_layout(nx_graph))
                plt.savefig(args.res_dir[type_c] +"/pic_"+str(idx)+".pdf")
                plt.close()

    for graph in graph_predict:
        for node in graph.vs:
            node["type"] = event_types_ontology_new[node["type"]]


    for j, tg in enumerate(g_list_test):
        f_event = 0
        f_s2, f_s3 = 0, 0
        for i, g in enumerate(graph_predict):
            _, _, f_e = calcu_event_match(g, tg)
            f_event += f_e
            f2, f3 = calcu_seq_match(g, tg)
            f_s2 += f2
            f_s3 += f3
        f_event /= len(graph_predict)
        f_s2 /= len(graph_predict)
        f_s3 /= len(graph_predict)
        # print(len(tg.vs), f_event, f_s2, f_s3)
    
    _, _, x = node_and_edge_match(graph_predict, g_list_test)
    kl_and_MCS(graph_predict, g_list_test)
    return x


def evaluation_generation_res(file_name, args, type_c = None, seed = 6666):
    print("*"*20 + file_name + "*"*20)
    res_dir = "../result_"+args.data_type+"/dataset_{}_seed_{}".format(file_name, seed) + "/predict_igraph.pkl"
    g_list_test = []
    
    event_types_ontology_new = _load_event_ontology()

    print("../../data/Wiki_IED_split/test/" + file_name + '_test_human_pruned_new_no_iso_max_50_igraphs.pkl')
    # 读取测试数据
    [graphs_test ,depth] = pickle.load(
        open("../../data/Wiki_IED_split/test/" + file_name + '_test_human_pruned_new_no_iso_max_50_igraphs.pkl', 'rb'))
   
    graphs_test = [graphs_test]
    for mm in range(len(graphs_test)):
        for i, row in enumerate(graphs_test[mm]):
            g, n = row   
            g_list_test.append(g)
    graph_predict = pickle.load(open(res_dir, 'rb'))
    print(len(graph_predict))

    
    if type_c != None:   
        for idx, g in enumerate(graph_predict):
            nx_graph = nx.DiGraph()
            node_id_to_type = {}
            all_edges = list(g.es)
            for ii in range(len(all_edges)):
                node_id_to_type[all_edges[ii].source] = g.vs[all_edges[ii].source]['type'].split(".")[1] if g.vs[all_edges[ii].source]['type'].split(".")[-1] == 'Unspecified' else g.vs[all_edges[ii].source]['type'].split(".")[-1]
                node_id_to_type[all_edges[ii].target] = g.vs[all_edges[ii].target]['type'].split(".")[1] if g.vs[all_edges[ii].target]['type'].split(".")[-1] == 'Unspecified' else g.vs[all_edges[ii].target]['type'].split(".")[-1]
                nx_graph.add_edge(all_edges[ii].source, all_edges[ii].target)
                nx.draw_networkx(
                    nx_graph, with_labels=True, labels=node_id_to_type, font_size=3,
                    node_size=50, arrowsize=3, width=0.5, pos=nx.spring_layout(nx_graph))
                plt.savefig(args.res_dir[type_c] +"/pic_"+str(idx)+".pdf")
                plt.close()

    for graph in graph_predict:
        for node in graph.vs:
            node["type"] = event_types_ontology_new[node["type"]]

    # 进行评测

    for j, tg in enumerate(g_list_test):
        f_event = 0
        f_s2, f_s3 = 0, 0
        for i, g in enumerate(graph_predict):
            _, _, f_e = calcu_event_match(g, tg)
            f_event += f_e
            f2, f3 = calcu_seq_match(g, tg)
            f_s2 += f2
            f_s3 += f3
        f_event /= len(graph_predict)
        f_s2 /= len(graph_predict)
        f_s3 /= len(graph_predict)
        # print(len(tg.vs), f_event, f_s2, f_s3)
    
    # _, _, x = node_and_edge_match(graph_predict, g_list_test)
    # kl_and_MCS(graph_predict, g_list_test)
    _, _, x = node_and_edge_match(graph_predict, [g for g in g_list_test if len(g.vs)>=1])
    kl_and_MCS(graph_predict, [g for g in g_list_test if len(g.vs)>=1])
    return x

if __name__ == "__main__":
    # 42, 0, 1, 2, 3, 5, 6, 71, 72, 81, 82, 83, 91, 92
    for seed in [42]:
        print(seed)
        parser = argparse.ArgumentParser(
            description='using diffusion model to generate complex event schema')
        parser.add_argument("--data_type", type=str, default="IGE")
        args = parser.parse_args()
        print("suicide_ied")
        evaluation_generation_res('suicide_ied', args, seed=seed)
        print("wiki_ied_bombings")
        evaluation_generation_res('wiki_ied_bombings', args, seed=seed)
        print("wiki_mass_car_bombings")
        evaluation_generation_res('wiki_mass_car_bombings', args, seed=seed)
        print("\n\n\n")

    # seed = 42
    # parser = argparse.ArgumentParser(
    #     description='using diffusion model to generate complex event schema')
    # parser.add_argument("--data_type", type=str, default="None")
    # args = parser.parse_args()
    # # print("suicide_ied")
    # # evaluation_generation_res_on_origin_test_dataset('suicide_ied', args, seed=seed)
    # # print("wiki_ied_bombings")
    # # evaluation_generation_res_on_origin_test_dataset('wiki_ied_bombings', args, seed=seed)
    # # print("wiki_mass_car_bombings")
    # # evaluation_generation_res_on_origin_test_dataset('wiki_mass_car_bombings', args, seed=seed)

    # print("suicide_ied")
    # evaluation_generation_res('suicide_ied', args, seed=seed)
    # print("wiki_ied_bombings")
    # evaluation_generation_res('wiki_ied_bombings', args, seed=seed)
    # print("wiki_mass_car_bombings")
    # evaluation_generation_res('wiki_mass_car_bombings', args, seed=seed)
