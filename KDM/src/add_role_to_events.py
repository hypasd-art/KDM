import pandas as pd
import pickle
import numpy as np
import torch.nn.functional as F
import torch
import igraph
import operator

def is_nan(nan):
    return nan != nan

def filter_relations0(dataset):
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = dataset
    
    all_pruned_graph_event_entity_relations = []
    all_pruned_graph_entity_relations = []
    all_pruned_graph_events = []
    all_pruned_graph_entities = []
    
    for ii in range(len(all_graph_event_links)):
        curr_graph_pruned_event_links = all_graph_event_links[ii]
        curr_graph_event_entity_relations = all_graph_event_entity_relations[ii]
        curr_graph_entity_relations = all_graph_entity_relations[ii]
        curr_graph_mentioned_events = set()
        
        for jj in range(len(curr_graph_pruned_event_links)):
            curr_graph_mentioned_events.add(curr_graph_pruned_event_links[jj][0])
            curr_graph_mentioned_events.add(curr_graph_pruned_event_links[jj][1])
        curr_graph_pruned_events = []
        curr_graph_events = all_graph_events[ii]
        
        for jj in range(len(curr_graph_events)):
            if curr_graph_events[jj][0] in curr_graph_mentioned_events:
                curr_graph_pruned_events.append(curr_graph_events[jj])
        curr_mentioned_entities = set()
        curr_graph_pruned_event_entity_relations = []
        
        for jj in range(len(curr_graph_event_entity_relations)):
            temp_event = curr_graph_event_entity_relations[jj][0]
            if temp_event not in curr_graph_mentioned_events:
                continue
            curr_mentioned_entities.add(curr_graph_event_entity_relations[jj][2])
            curr_graph_pruned_event_entity_relations.append(
                curr_graph_event_entity_relations[jj])
        curr_graph_pruned_entity_relations = []
        
        for jj in range(len(curr_graph_entity_relations)):
            temp_entity_start = curr_graph_entity_relations[jj][0]
            temp_entity_end = curr_graph_entity_relations[jj][2]
            if (temp_entity_start not in curr_mentioned_entities or
                temp_entity_end not in curr_mentioned_entities):
                continue
            curr_graph_pruned_entity_relations.append(curr_graph_entity_relations[jj])
        curr_graph_pruned_entities = []
        curr_graph_entities = all_graph_entities[ii]
        
        for jj in range(len(curr_graph_entities)):
            if curr_graph_entities[jj][0] in curr_mentioned_entities:
                curr_graph_pruned_entities.append(curr_graph_entities[jj])
        all_pruned_graph_event_entity_relations.append(
            curr_graph_pruned_event_entity_relations)
        all_pruned_graph_entity_relations.append(curr_graph_pruned_entity_relations)
        all_pruned_graph_events.append(curr_graph_pruned_events)
        all_pruned_graph_entities.append(curr_graph_pruned_entities)
    return [all_pruned_graph_events, 
        all_pruned_graph_event_entity_relations,
        all_graph_event_links, 
        all_pruned_graph_entities,
        all_pruned_graph_entity_relations]

def calcu_role_info(scenario):
    pruned_file = '../../data/Wiki_IED_split/train/' + scenario +'_train_pruned_new_no_iso_max_50_dataset.pkl'
    dataset = pickle.load(open(pruned_file, "rb"))
    dict_role = {}

    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = filter_relations0(dataset)


    for ii in range(len(all_graph_events)):
        events = all_graph_events[ii]
        event_entity_relas = all_graph_event_entity_relations[ii]
        event_links = all_graph_event_links[ii]
        entities = all_graph_entities[ii]
        entity_relas = all_graph_entity_relations[ii]

        entity_type_dict = {}
        event_type_dict = {}
        for ii in range(len(events)):
            event_type_dict[events[ii][0]] = events[ii][1]
        for ii in range(len(entities)):
            entity_type_dict[entities[ii][0]] = entities[ii][1]
        for ii in range(len(event_entity_relas)):
            event_id, rela_type, entity_id = event_entity_relas[ii]
            entity_type = entity_type_dict[entity_id]
            event_type = event_type_dict[event_id]
            if event_type not in dict_role:
                dict_role[event_type] = {}
            if rela_type not in dict_role[event_type]:
                dict_role[event_type][rela_type] = {}
                dict_role[event_type][rela_type][entity_type] = 1
            else:
                if entity_type not in dict_role[event_type][rela_type]:
                    dict_role[event_type][rela_type][entity_type] = 1
                else:
                    dict_role[event_type][rela_type][entity_type] += 1
    # print(dict_role)
    return dict_role

def _load_event_ontology(file_path = "../../data/kairos-ontology.xlsx"):
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    event_entity_rel_dict = {}

    event_types_ontology = saved_dict['event_types']
    event_entity_relation_ontology = saved_dict['event_entity_relations']

    event_types_ontology_new = {0: "START", 1: "END"}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[val + 2] = key

    ontology = pd.read_excel(file_path, sheet_name="events")
    event_types = ontology['Type'].tolist()
    event_subtypes = ontology['Subtype'].tolist()
    event_sub_subtypes = ontology['Sub-subtype'].tolist()
    arg1 = ontology['arg1 label'].tolist()
    arg2 = ontology['arg2 label'].tolist()
    arg3 = ontology['arg3 label'].tolist()
    arg4 = ontology['arg4 label'].tolist()
    arg5 = ontology['arg5 label'].tolist()
    arg6 = ontology['arg6 label'].tolist()

    con1 = ontology['arg1 type constraints'].tolist()
    con2 = ontology['arg2 type constraints'].tolist()
    con3 = ontology['arg3 type constraints'].tolist()
    con4 = ontology['arg4 type constraints'].tolist()
    con5 = ontology['arg5 type constraints'].tolist()
    con6 = ontology['arg6 type constraints'].tolist()

    for ii in range(len(event_types)):
        assert event_types_ontology[event_types[ii] + "." + event_subtypes[ii] +
            '.' + event_sub_subtypes[ii]] == ii
        role_list = []
        if not is_nan(arg1[ii]):
            con_s = con1[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg1[ii], con_s[0].strip()])
        if not is_nan(arg2[ii]):
            con_s = con2[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg2[ii], con_s[0].strip()])
        if not is_nan(arg3[ii]):
            con_s = con3[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg3[ii], con_s[0].strip()])
        if not is_nan(arg4[ii]):
            con_s = con4[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg4[ii], con_s[0].strip()])
        if not is_nan(arg5[ii]):
            con_s = con5[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg5[ii], con_s[0].strip()])
        if not is_nan(arg6[ii]):
            con_s = con6[ii].split(",")
            if "event" in con_s:
                con_s.remove("event")
            if len(con_s) != 0:
                role_list.append([arg6[ii], con_s[0].strip()])
        event_entity_rel_dict[event_types[ii] + "." + event_subtypes[ii] +
            '.' + event_sub_subtypes[ii]] = role_list

    return event_types_ontology_new, event_entity_rel_dict

def add_role(file_name, seed = 6666):
    role_dict = calcu_role_info(file_name)
    res_dir = "../result_IGE/dataset_{}_seed_{}".format(file_name, seed) + "/predict_igraph.pkl"
    graph_predict = pickle.load(open(res_dir, 'rb'))
    all_graph_events = []
    all_graph_event_entity_relations = []
    all_graph_event_links = []
    all_graph_entities = []
    all_graph_event_cluster = []
    for graph in graph_predict:
        while True:
            judge = False
            for i in range(len(graph.vs)):
                node = graph.vs[i]
                if node['type'] == "START" or node['type'] == "END":
                    graph.delete_vertices(node)
                    judge = True
                    break
            if judge == False:
                break
        num = len(graph.vs)

        all_events  = []
        all_event_entity_relations = []
        all_entities = []
        all_event_links = []
        all_event_cluster = {}
        for i, node in enumerate(graph.vs):
            all_events.append((i, node['type']))
            roles = role_dict[node['type']]
            event_cluster = []
            for rel, role in roles.items():
                event_cluster.append(num)
                all_event_entity_relations.append((i, rel, num))
                all_entities.append((num, max(role.items(), key=operator.itemgetter(1))[0]))
                num += 1
            all_event_cluster[i] = event_cluster

        for edge_tuple in graph.get_edgelist():
            all_event_links.append((edge_tuple[0], edge_tuple[1]))
        all_graph_events.append(all_events)
        all_graph_event_entity_relations.append(all_event_entity_relations)
        all_graph_event_links.append(all_event_links)
        all_graph_entities.append(all_entities)
        all_graph_event_cluster.append(all_event_cluster)
     

    with open("../result_IGE/dataset_{}_seed_{}".format(file_name, seed)+'/predict_add_roles_dataset.pkl', 'wb') as handle:
        pickle.dump([all_graph_events, 
            all_graph_event_entity_relations,
            all_graph_event_links, 
            all_graph_entities,
            all_graph_event_cluster], handle)
    print("add roles done")

def _load_ontology_dict():
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    event_types = saved_dict["event_types"]
    event_entity_relations = saved_dict["event_entity_relations"]
    entity_types_dict = saved_dict["entity_types_dict"]
    entity_relation_dict = saved_dict["entity_relation_dict"]

    node_type_dict = {}
    for key, val in event_types.items():
        node_type_dict[key] = val
    for key, val in entity_types_dict.items():
        node_type_dict[key] = val + 67

    edge_type_dict = {"event-event-before-after": 0, "no-edge": 1,
        "identical-edge": 2}
    for key, val in event_entity_relations.items():
        edge_type_dict[key] = val + 3
    for key, val in entity_relation_dict.items():
        edge_type_dict[key] = val + 3 + 85

    return node_type_dict, edge_type_dict          
            
def _load_reverse_ontology_dict():
    saved_dict = pickle.load(open("../../data/kairos_ontology.pkl", "rb"))[0]
    event_types = saved_dict["event_types"]
    event_entity_relations = saved_dict["event_entity_relations"]
    entity_types_dict = saved_dict["entity_types_dict"]
    entity_relation_dict = saved_dict["entity_relation_dict"]

    node_type_dict_reverse = {}
    for key, val in event_types.items():
        node_type_dict_reverse[val] = key
    for key, val in entity_types_dict.items():
        node_type_dict_reverse[val+67] = key

    edge_type_dict_reverse = {0:"event-event-before-after", 1:"no-edge",2:"identical-edge"}
    for key, val in event_entity_relations.items():
        edge_type_dict_reverse[val+3] = key
    for key, val in entity_relation_dict.items():
        edge_type_dict_reverse[val + 3 + 85] = key

    return node_type_dict_reverse, edge_type_dict_reverse

def filter_relations(dataset):
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_event_cluster] = dataset
    
    all_pruned_graph_event_entity_relations = []
    all_pruned_graph_events = []
    all_pruned_graph_entities = []
    
    for ii in range(len(all_graph_event_links)):
        curr_graph_pruned_event_links = all_graph_event_links[ii]
        curr_graph_event_entity_relations = all_graph_event_entity_relations[ii]
        curr_graph_mentioned_events = set()
        
        for jj in range(len(curr_graph_pruned_event_links)):
            curr_graph_mentioned_events.add(curr_graph_pruned_event_links[jj][0])
            curr_graph_mentioned_events.add(curr_graph_pruned_event_links[jj][1])
        curr_graph_pruned_events = []
        curr_graph_events = all_graph_events[ii]
        
        for jj in range(len(curr_graph_events)):
            if curr_graph_events[jj][0] in curr_graph_mentioned_events:
                curr_graph_pruned_events.append(curr_graph_events[jj])
        curr_mentioned_entities = set()
        curr_graph_pruned_event_entity_relations = []
        
        for jj in range(len(curr_graph_event_entity_relations)):
            temp_event = curr_graph_event_entity_relations[jj][0]
            if temp_event not in curr_graph_mentioned_events:
                continue
            curr_mentioned_entities.add(curr_graph_event_entity_relations[jj][2])
            curr_graph_pruned_event_entity_relations.append(
                curr_graph_event_entity_relations[jj])
        curr_graph_pruned_entities = []
        curr_graph_entities = all_graph_entities[ii]
        
        for jj in range(len(curr_graph_entities)):
            if curr_graph_entities[jj][0] in curr_mentioned_entities:
                curr_graph_pruned_entities.append(curr_graph_entities[jj])
        all_pruned_graph_event_entity_relations.append(curr_graph_pruned_event_entity_relations)
        all_pruned_graph_events.append(curr_graph_pruned_events)
        all_pruned_graph_entities.append(curr_graph_pruned_entities)
    return [all_pruned_graph_events, 
        all_pruned_graph_event_entity_relations,
        all_graph_event_links, 
        all_pruned_graph_entities,
        all_graph_event_cluster]

def _construct_graph_init(events, entities, event_entity_relas, event_links):
    num_nodes = len(events) + len(event_entity_relas)
    
    g = igraph.Graph(directed=False)
    g.add_vertices(num_nodes)
    
    node_dict = {}
    x_features = []
    node_name = []
    role_feature = []
    
    A_init = np.zeros((num_nodes, num_nodes, len(edge_type_ontology)))
    dis_matrix = np.zeros((num_nodes, num_nodes))
    A_init[:, :, 1] = 1 
    
    entity_type_dict = {}
    event_type_dict = {}
    for ii in range(len(events)):
        event_type_dict[events[ii][0]] = events[ii][1]
    for ii in range(len(entities)):
        entity_type_dict[entities[ii][0]] = entities[ii][1]


    events_role_set = {}
    for ii in range(len(events)):
        event_id, event_type = events[ii]
        events_role_set[event_id] = []
        node_dict[event_id] = set([ii])
        temp_feature = np.append(node_type_embeddings[node_type_ontology[event_type]], node_type_embeddings[node_type_ontology[event_type]])
        node_name.append(event_type)
        x_features.append(temp_feature)
        role_feature.append(1)
    assert len(x_features) == len(events)

    for ii in range(len(event_links)):
        start_event, end_event = event_links[ii]
        start_id = list(node_dict[start_event])[0]
        assert len(list(node_dict[start_event])) == 1
        assert len(list(node_dict[end_event])) == 1
        end_id = list(node_dict[end_event])[0]
        g.add_edge(start_id, end_id)
        A_init[start_id, end_id, 1] = 0
        A_init[start_id, end_id, 0] = 1
        A_init[end_id, start_id, 1] = 0
        A_init[end_id, start_id, 0] = 1
    
    for ii in range(len(event_entity_relas)):
        event_id, rela_type, entity_id = event_entity_relas[ii]
        events_role_set[event_id].append(ii + len(events))
        if entity_id in node_dict:
            node_dict[entity_id].add(ii + len(events))
        else:
            node_dict[entity_id] = set([ii + len(events)])
        entity_type = entity_type_dict[entity_id]
        event_type = event_type_dict[event_id]
        temp_feature = np.append(node_type_embeddings[node_type_ontology[entity_type.upper()]],node_type_embeddings[node_type_ontology[event_type]])
        x_features.append(temp_feature)
        role_feature.append(edge_type_ontology[rela_type])
        node_name.append(entity_type.upper())


        A_init[list(node_dict[event_id])[0], ii + len(events), 1] = 0
        A_init[list(node_dict[event_id])[0], ii + len(events), edge_type_ontology[rela_type]] = 1
        A_init[ii + len(events), list(node_dict[event_id])[0], 1] = 0
        A_init[ii + len(events), list(node_dict[event_id])[0], edge_type_ontology[rela_type]] = 1
        g.add_edge(list(node_dict[event_id])[0], ii + len(events))
    for ii in range(len(g.vs)):
        for jj in range(len(g.vs)):
            length = len(g.get_shortest_paths(ii, to=jj)[0])
            dis_matrix[ii][jj] = length
    assert len(x_features) == len(events) + len(event_entity_relas)
        
    return A_init, node_dict, g, x_features, entity_type_dict, len(events), node_name, dis_matrix, events_role_set, role_feature

def _convert_A_to_dense(matrix):
    ret_mat = np.zeros((len(matrix), len(matrix)))
    for ii in range(len(matrix)):
        for jj in range(len(matrix[ii])):
            if len(np.nonzero(matrix[ii][jj])[0]) != 1:
                print(np.nonzero(matrix[ii][jj]))
                raise
            ret_mat[ii, jj] = int(np.nonzero(matrix[ii][jj])[0])
    return ret_mat

def _construct_graph(pruned_file):
    dataset = pickle.load(open(pruned_file, "rb"))
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_event_cluster] = filter_relations(dataset)

    all_A_init = []
    all_x_features = []
    all_event_len = []
    all_dis = []
    all_node_dict = []
    all_e_r_dict = []
    all_role_f = []

    for ii in range(len(all_graph_events)):
        events = all_graph_events[ii]
        event_entity_relas = all_graph_event_entity_relations[ii]
        event_links = all_graph_event_links[ii]
        entities = all_graph_entities[ii]

        A_init, node_dict, g, X_features, entity_type_dict, len_event, node_name, dis_matrix, events_role_set, role_f = _construct_graph_init(
            events, entities, event_entity_relas, event_links)

        if np.sum(A_init) != A_init.shape[0] * A_init.shape[1]:
            print("init not qeual")
            raise

        A_init_dense = _convert_A_to_dense(A_init)

        all_A_init.append(A_init_dense)
        all_x_features.append(np.array(X_features))
        all_event_len.append(len_event)
        all_node_dict.append(node_name)
        all_dis.append(dis_matrix)
        all_e_r_dict.append(events_role_set)
        all_role_f.append(role_f)

    return all_A_init, all_x_features, all_event_len, all_node_dict, all_dis, all_e_r_dict, all_role_f

def read_dataset(scenario, seed = 6666):
    A_init, x_features, train_len, all_node_dict, dis, e_r, all_role = _construct_graph(
        "../result_IGE/dataset_{}_seed_{}".format(scenario, seed)+'/predict_add_roles_dataset.pkl')
    return A_init, x_features, train_len, all_node_dict, dis, e_r, all_role

def visualization(args, scenario, all_node_dict, e_r):
    node_type_dict_reverse, edge_type_dict_reverse = _load_reverse_ontology_dict()
    rel_list = pickle.load(
        open("../result_IGE/dataset_{}_seed_{}".format(scenario, args.seed)+'/predict_rel_final.pkl', 'rb'))
    rels= []
    for idx, rel in enumerate(rel_list):
        print(idx)
        rel = torch.argmax(rel.squeeze(0), dim=-1).cpu().numpy()
        e_r_idx = e_r[idx]
        for k, v in e_r_idx.items():
            for i in v:
                for j in v:
                    rel[i][j] = 1
                    rel[j][i] = 1
        print((rel != 1).sum(), rel.shape[0]*rel.shape[1])
        rels.append(rel)
    assert len(rels) == len(all_node_dict)
    with open("../result_IGE/dataset_{}_seed_{}".format(scenario, args.seed)+'/to_visualization.pkl', 'wb') as handle:
        pickle.dump([all_node_dict, rels, edge_type_dict_reverse], handle)

node_type_embeddings = pickle.load(open(
        "../../data/kairos_ontology_embeddings.pkl", "rb"))
event_types_ontology_new, event_entity_rel_dict= _load_event_ontology()
node_type_ontology, edge_type_ontology = _load_ontology_dict()
if __name__ == "__main__":
    
    print("suicide_ied")
    add_role('suicide_ied')
    read_dataset("suicide_ied")
    print("wiki_ied_bombings")
    add_role('wiki_ied_bombings')
    read_dataset('wiki_ied_bombings')
    print("wiki_mass_car_bombings")
    add_role('wiki_mass_car_bombings')
    read_dataset('wiki_mass_car_bombings')


