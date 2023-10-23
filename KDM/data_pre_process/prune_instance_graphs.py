"""
The code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import json
import numpy as np
import pickle
import pandas as pd
import igraph
import networkx as nx
import matplotlib.pyplot as plt
max_node = 50
print("max_node_num = " + str(max_node))

def _load_event_ontology():
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {"START": 0, "END": 1}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[key] = val + 2
    return event_types_ontology_new


def _construct_event_name_to_id_dict(all_graph_events, events_ontology):
    event_dict = {}
    for ii in range(len(all_graph_events)):
        for jj in range(len(all_graph_events[ii])):
            curr_entity = all_graph_events[ii][jj]
            event_dict[curr_entity[0]] = events_ontology[curr_entity[1]]
    return event_dict


def _get_graph_node_num(graph_list):
    all_nodes = set()
    graph_node_ind_dict = {}
    graph_all_nodes = nx.DiGraph()
    for ii in range(len(graph_list)):
        curr_start = graph_list[ii][0]
        curr_end = graph_list[ii][1]

        graph_all_nodes.add_edge(curr_start, curr_end)

    graph_without_iso = graph_all_nodes.copy()
    isolated_events = list(nx.isolates(graph_without_iso))
    graph_without_iso.remove_nodes_from(isolated_events)

    graph_all_nodes_list = list(graph_all_nodes.nodes)
    graph_all_nodes_name_to_id_dict = {}
    for ii in range(len(graph_all_nodes_list)):
        graph_all_nodes_name_to_id_dict[graph_all_nodes_list[ii]] = ii + 1

    graph_without_iso_list = list(graph_without_iso.nodes)
    graph_without_iso_name_to_id_dict = {}
    for ii in range(len(graph_without_iso_list)):
        graph_without_iso_name_to_id_dict[graph_without_iso_list[ii]] = ii + 1
    return graph_all_nodes_name_to_id_dict, graph_without_iso_name_to_id_dict


def _get_nodes_for_start_end(graph, node_num):
    indegree_0 = []
    outdegree_0 = []
    for ii in range(1, node_num + 1):
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



def _construct_and_reorder(graph_name_to_id, event_dict, event_links):
    g = igraph.Graph(directed=True)
    n = len(graph_name_to_id)
    g.add_vertices(n + 2)
    g.vs[0]['type'] = 0
    g.vs[n + 1]['type'] = 1
    for ii in range(len(event_links)):
        if (event_links[ii][0] not in graph_name_to_id or
            event_links[ii][1] not in graph_name_to_id):
            continue

        start_node_ind = graph_name_to_id[event_links[ii][0]]
        start_node_type = event_dict[event_links[ii][0]]

        end_node_ind = graph_name_to_id[event_links[ii][1]]
        end_node_type = event_dict[event_links[ii][1]]

        g.vs[start_node_ind]['type'] = start_node_type
        g.vs[end_node_ind]['type'] = end_node_type

        g.add_edge(start_node_ind, end_node_ind)

    indegree_0, outdegree_0 = _get_nodes_for_start_end(g, len(graph_name_to_id))

    for ii in range(1, n + 1):
        if ii in indegree_0:
            g.add_edge(0, ii)
        if ii in outdegree_0:
            g.add_edge(ii, n + 1)

    g = _reorder_graph(g)
    return g




def _construct_event_graph(all_graph_event_links, event_dict, all_graph_events, plt_name):
    all_igraphs_all_nodes = []
    all_igraphs_no_iso = []
    all_len = []
    valid_ind = []
    for ii in range(len(all_graph_event_links)):
        curr_graph = all_graph_event_links[ii]
        graph_all_name_to_id, graph_no_iso_name_to_id = _get_graph_node_num(curr_graph)


        g_all = _construct_and_reorder(graph_all_name_to_id, event_dict, curr_graph)
        g_no_iso = _construct_and_reorder(graph_no_iso_name_to_id, event_dict, curr_graph)


        if len(list(g_all.vs)) > max_node:
            continue
        valid_ind.append(ii)
        
        all_igraphs_all_nodes.append((g_all, len(list(g_all.vs))))
        all_igraphs_no_iso.append((g_no_iso, len(list(g_no_iso.vs))))

        if len(list(g_all.vs)) > len(list(g_no_iso.vs)):
            print("not equal")
    return all_igraphs_all_nodes, all_igraphs_no_iso, valid_ind


def _remove_same_type_links(event_links, event_name_to_id):
    all_valid_event_links = set()
    for ii in range(len(event_links)):
        start, end = event_links[ii]
        if event_name_to_id[start] == event_name_to_id[end]:
            for jj in range(len(event_links)):
                if event_links[jj][0] == end:
                    all_valid_event_links.add((start, event_links[jj][1]))
                    if event_links[jj] in all_valid_event_links:
                        all_valid_event_links.remove(event_links[jj])
                    if event_links[ii] in all_valid_event_links:
                        all_valid_event_links.remove(event_links[ii])
        else:
            all_valid_event_links.add(event_links[ii])
    return list(all_valid_event_links)

def _remove_same_edge(event_links, event_name_to_id):
    start_dict = {}
    end_dict = {}
    for ii in range(len(event_links)):
        if event_links[ii][0] in start_dict:
            start_dict[event_links[ii][0]].add(event_links[ii][1])
        else:
            start_dict[event_links[ii][0]] = set()
            start_dict[event_links[ii][0]].add(event_links[ii][1])

    all_valid_event_links = set(event_links)
    for key, val in start_dict.items():
        curr_list = list(val)
        curr_merge_start = curr_list[0]
        for ii in range(len(event_links)):
            if event_links[ii][0] != curr_merge_start and event_links[ii][0] in val:
                if event_links[ii] in all_valid_event_links:
                    all_valid_event_links.remove(event_links[ii])
                all_valid_event_links.add((curr_merge_start, event_links[ii][1]))
        for ii in range(1, len(curr_list)):
            if (key, curr_list[ii]) in all_valid_event_links:
                all_valid_event_links.remove((key, curr_list[ii]))

    return list(all_valid_event_links)


def _convert_event_entity_link_to_ids(
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_events, 
    event_name_to_id,
    all_graph_entity_relations):
    all_valid_event_links = []
    for ii in range(len(all_graph_event_entity_relations)):
        event_entity_shared_dict = {}
        curr_graph_event_entity_links = all_graph_event_entity_relations[ii]
        curr_graph_event_links = all_graph_event_links[ii]
        curr_graph_events = all_graph_events[ii]
        for jj in range(len(curr_graph_event_entity_links)):
            event_entity_link = curr_graph_event_entity_links[jj]
            if event_entity_link[2] in event_entity_shared_dict:
                event_entity_shared_dict[event_entity_link[2]].add(event_entity_link[0])
            else:
                event_entity_shared_dict[event_entity_link[2]] = set()
                event_entity_shared_dict[event_entity_link[2]].add(event_entity_link[0])
        
        curr_entity_entity_dict = {}
        curr_graph_entity_entity_links = all_graph_entity_relations[ii]
        for jj in range(len(curr_graph_entity_entity_links)):
            entity_start, _, entity_end = curr_graph_entity_entity_links[jj]
            if entity_start in curr_entity_entity_dict:
                curr_entity_entity_dict[entity_start].add(entity_end)
            else:
                curr_entity_entity_dict[entity_start] = set()
                curr_entity_entity_dict[entity_start].add(entity_end)
            if entity_end in curr_entity_entity_dict:
                curr_entity_entity_dict[entity_end].add(entity_start)
            else:
                curr_entity_entity_dict[entity_end] = set()
                curr_entity_entity_dict[entity_end].add(entity_start)
        # print(curr_entity_entity_dict)
        for key, val in event_entity_shared_dict.items():
            val_lst = list(val)
            for jj in range(len(val_lst)):
                if val_lst[jj] in curr_entity_entity_dict:
                    # print(val_lst[jj])
                    val.update(curr_entity_entity_dict[val_lst[jj]])

        valid_event_links = []
        for jj in range(len(curr_graph_event_links)):
            start, end = curr_graph_event_links[jj]
            for key, val in event_entity_shared_dict.items():
                if start in val and end in val:
                    valid_event_links.append((start, end))

        valid_event_links = _remove_same_type_links(valid_event_links,event_name_to_id)
        all_valid_event_links.append(valid_event_links)
    return all_valid_event_links


from igraph import *

def get_node_depth(graphs):
    path_depth = []
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
            path = g.get_shortest_paths(start, to=i, output='vpath')
            graph_depth.append(len(path[0]))
        path_depth.append(graph_depth)
    return path_depth

def construct_event_igraphs(file_name, plt_name):
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = pickle.load(open(file_name + ".pkl", "rb"))
    events_ontology = _load_event_ontology()
    # {event_name: event_type_id}
    event_name_to_id = _construct_event_name_to_id_dict(all_graph_events, events_ontology)

    all_valid_event_links = _convert_event_entity_link_to_ids(
        all_graph_event_entity_relations, 
        all_graph_event_links, 
        all_graph_events,
        event_name_to_id, 
        all_graph_entity_relations)

    assert len(all_valid_event_links) == len(all_graph_event_links)
    print(file_name + " the number of event nodes"  + " is " + str(len(all_graph_events)))

    event_graphs_all, event_graphs_no_iso, valid_ind = _construct_event_graph(
        all_valid_event_links, 
        event_name_to_id, 
        all_graph_events, 
        plt_name)
    
    event_depth = get_node_depth(event_graphs_no_iso)

    all_updated_graph_events = []
    all_updated_graph_event_entity_relations = []
    all_updated_graph_event_links = []
    all_updated_graph_entities = []
    all_updated_graph_entity_relations = []
    assert len(valid_ind) == len(event_graphs_no_iso)
    for ind in valid_ind:
        all_updated_graph_events.append(all_graph_events[ind])
        all_updated_graph_event_entity_relations.append(all_graph_event_entity_relations[ind])
        all_updated_graph_event_links.append(all_valid_event_links[ind])
        all_updated_graph_entities.append(all_graph_entities[ind])
        all_updated_graph_entity_relations.append(all_graph_entity_relations[ind])


    print(file_name + " the number of graph with event nodes < " + str(max_node) + " is " + str(len(event_graphs_no_iso)))

    with open(file_name + '_pruned_new_no_iso_max_'+ str(max_node) + '_igraphs.pkl', 'wb') as handle:
        pickle.dump([event_graphs_no_iso, event_depth], handle)

    with open(file_name + '_pruned_new_no_iso_max_'+ str(max_node) + '_dataset.pkl', 'wb') as handle:
        pickle.dump([
            all_updated_graph_events, 
            all_updated_graph_event_entity_relations,
            all_updated_graph_event_links, 
            all_updated_graph_entities,
            all_updated_graph_entity_relations], handle)



if __name__ == "__main__":
    file_dir = "../../data/Wiki_IED_split/"
    dataset_types = ["train", "dev", "test"]
    file_names = ["suicide_ied_", "wiki_ied_bombings_", "wiki_mass_car_bombings_"]
    
    for file_name in file_names:
        for dataset_type in dataset_types:
            curr_file = file_dir + dataset_type + "/" + file_name + dataset_type
            construct_event_igraphs(curr_file, file_name + dataset_type)









































