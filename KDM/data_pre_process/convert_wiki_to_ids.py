"""
The code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import numpy as np
import pickle
import json
import pandas as pd
import igraph

def _load_ontology():
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    return saved_dict


# return dict
# {event_name, event_type}
# ('caci:Schemas/Instantiated/cluster_429/Steps/EN_Event_0013306', 'Transaction.ExchangeBuySell.Unspecified')
def _construct_event_dict(all_graph_events):
    event_dict = {}
    for ii in range(len(all_graph_events)):
        for jj in range(len(all_graph_events[ii])):
            curr_entity = all_graph_events[ii][jj]
            event_dict[curr_entity[0]] = curr_entity[1]
            # print(curr_entity)
    return event_dict

def _construct_entity_dict(all_graph_entities):
    entity_dict = {}
    for ii in range(len(all_graph_entities)):
        for jj in range(len(all_graph_entities[ii])):
            curr_entity = all_graph_entities[ii][jj]
            entity_dict[curr_entity[0]] = curr_entity[1]
    return entity_dict

# return tuple (id, id)
def _convert_event_types_to_ids(all_graph_event_links, event_types_ontology, event_to_type_dict):
    event_links_results = []
    for ii in range(len(all_graph_event_links)):
        curr_graph = all_graph_event_links[ii]
        curr_graph_results = []
        for jj in range(len(curr_graph)):
            start_id = event_types_ontology[event_to_type_dict[curr_graph[jj][0]]]
            end_id = event_types_ontology[event_to_type_dict[curr_graph[jj][1]]]
            curr_graph_results.append((start_id, end_id))
        event_links_results.append(curr_graph_results)
    return event_links_results

def _get_graph_node_num(graph_list):
    all_nodes = set()
    graph_node_ind_dict = {}
    count = 1
    for ii in range(len(graph_list)):
        all_nodes.add(graph_list[ii][0])
        all_nodes.add(graph_list[ii][1])
        if graph_list[ii][0] not in graph_node_ind_dict:
            graph_node_ind_dict[graph_list[ii][0]] = count
            count += 1
        if graph_list[ii][1] not in graph_node_ind_dict:
            graph_node_ind_dict[graph_list[ii][1]] = count
            count += 1
    return len(all_nodes), graph_node_ind_dict

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
        new_g.add_edge(new_order_dict[curr_edge.source],
            new_order_dict[curr_edge.target])
    return new_g


def _construct_event_graph(all_graph_event_links, event_links_to_ids):
    all_igraphs = []
    for ii in range(len(all_graph_event_links)):
        curr_graph = all_graph_event_links[ii]
        n, graph_node_ind_dict = _get_graph_node_num(curr_graph)

        g = igraph.Graph(directed=True)
        g.add_vertices(n + 2)
        g.vs[0]['type'] = 0
        g.vs[n + 1]['type'] = 1

        for jj in range(len(curr_graph)):
            start_node_ind = graph_node_ind_dict[curr_graph[jj][0]]
            start_node_type = event_links_to_ids[ii][jj][0]
            end_node_ind = graph_node_ind_dict[curr_graph[jj][1]]
            end_node_type = event_links_to_ids[ii][jj][1]

            g.vs[start_node_ind]['type'] = start_node_type + 2
            g.vs[end_node_ind]['type'] = end_node_type + 2

            g.add_edge(start_node_ind, end_node_ind)

        indegree_0, outdegree_0 = _get_nodes_for_start_end(g, n)

        for jj in range(1, n + 1):
            if jj in indegree_0:
                g.add_edge(0, jj)
            if jj in outdegree_0:
                g.add_edge(jj, n + 1)
        g = _reorder_graph(g)
        all_igraphs.append((g, n + 2))
    return all_igraphs



def _convert_event_entity_links_to_ids(all_graph_event_entity_relations,
        event_to_type_dict, 
        entity_to_type_dict, 
        event_types_ontology,
        event_entity_rela_types_ontology, entity_types_ontology):
    event_entity_links_results = []
    for ii in range(len(all_graph_event_entity_relations)):
        curr_graph = all_graph_event_entity_relations[ii]
        curr_graph_results = []
        for jj in range(len(curr_graph)):
            start_id = event_types_ontology[event_to_type_dict[curr_graph[jj][0]]]
            rela_id = event_entity_rela_types_ontology[curr_graph[jj][1]]
            end_id = entity_types_ontology[entity_to_type_dict[curr_graph[jj][2]]]

            curr_graph_results.append((start_id, rela_id, end_id))
        event_entity_links_results.append(curr_graph_results)
    return event_entity_links_results


def _convert_entity_links_to_ids(all_graph_entity_relations,
        entity_to_type_dict, 
        entity_types_ontology,
        entity_relations_ontology):
    entity_links_results = []
    for ii in range(len(all_graph_entity_relations)):
        curr_graph = all_graph_entity_relations[ii]
        curr_graph_results = []
        for jj in range(len(curr_graph)):
            start_id = entity_types_ontology[entity_to_type_dict[curr_graph[jj][0]]]
            rela_id = entity_relations_ontology[curr_graph[jj][1]]
            end_id = entity_types_ontology[entity_to_type_dict[curr_graph[jj][2]]]
            curr_graph_results.append((start_id, rela_id, end_id))
        entity_links_results.append(curr_graph_results)
    return entity_links_results



def load_wiki_pkl(file_name):
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = pickle.load(open(file_name + ".pkl", "rb"))
    
    ontology_dict = _load_ontology()

    event_types_ontology = ontology_dict['event_types']
    event_entity_rela_types_ontology = ontology_dict['event_entity_relations']
    entity_types_ontology = ontology_dict['entity_types_dict']
    entity_relations_ontology = ontology_dict['entity_relation_dict']

    event_to_type_dict = _construct_event_dict(all_graph_events)

    event_links_to_ids = _convert_event_types_to_ids(all_graph_event_links, event_types_ontology, event_to_type_dict)
    event_graphs = _construct_event_graph(all_graph_event_links, event_links_to_ids)

    entity_to_type_dict = _construct_entity_dict(all_graph_entities)

    event_entity_links_to_ids = _convert_event_entity_links_to_ids(
        all_graph_event_entity_relations, 
        event_to_type_dict,
        entity_to_type_dict, 
        event_types_ontology,
        event_entity_rela_types_ontology, 
        entity_types_ontology)

    entity_entity_links_to_ids = _convert_entity_links_to_ids(
        all_graph_entity_relations, 
        entity_to_type_dict,
        entity_types_ontology, 
        entity_relations_ontology)
    with open(file_name + '_to_ids.pkl', 'wb') as handle:
        pickle.dump([event_links_to_ids, event_entity_links_to_ids,entity_entity_links_to_ids], handle)
    with open(file_name + '_igraphs.pkl', 'wb') as handle:
        pickle.dump(event_graphs, handle)


file_dir = "../../data/Wiki_IED_split/"
dataset_types = ["train", "dev", "test"]
file_names = ["suicide_ied_", "wiki_ied_bombings_", "wiki_mass_car_bombings_"]


for file_name in file_names:
    for dataset_type in dataset_types:
        curr_file = file_dir + dataset_type + "/" + file_name + dataset_type
        load_wiki_pkl(curr_file)

















