"""
Part of code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import igraph
import numpy as np
import pickle
import numpy as np

node_type_embeddings = pickle.load(open(
    "../../data/kairos_ontology_embeddings.pkl", "rb"))

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


node_type_ontology, edge_type_ontology = _load_ontology_dict()


def filter_relations(dataset):
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


def calcu_role_info(pruned_file):
    dataset = pickle.load(open(pruned_file, "rb"))
    dict_role = {}

    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = filter_relations(dataset)



    for ii in range(len(all_graph_events)):
        print("iter" + str(ii))
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
    print(dict_role)


def _construct_graph_init(events, entities, event_entity_relas, event_links):
    num_nodes = len(events) + len(event_entity_relas)
    
    g = igraph.Graph(directed=False)
    g.add_vertices(num_nodes)
    
    node_dict = {}
    node_dict_reverse = {}
    x_features = []
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

    for ii in range(len(events)):
        event_id, event_type = events[ii]
        node_dict[event_id] = set([ii])

        temp_feature = np.append(node_type_embeddings[node_type_ontology[event_type]], node_type_embeddings[node_type_ontology[event_type]])
        x_features.append(temp_feature)
        role_feature.append(edge_type_ontology["no-edge"])
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
        event_type = event_type_dict[event_id]
        if entity_id in node_dict:
            node_dict[entity_id].add(ii + len(events))
        else:
            node_dict[entity_id] = set([ii + len(events)])
        entity_type = entity_type_dict[entity_id]

        temp_feature = np.append(node_type_embeddings[node_type_ontology[entity_type]], node_type_embeddings[node_type_ontology[event_type]])
        x_features.append(temp_feature)
        role_feature.append(edge_type_ontology[rela_type])

        A_init[list(node_dict[event_id])[0], ii + len(events), 1] = 0
        A_init[list(node_dict[event_id])[0], ii + len(events), edge_type_ontology[rela_type]] = 1
        A_init[ii + len(events), list(node_dict[event_id])[0], 1] = 0
        A_init[ii + len(events), list(node_dict[event_id])[0], edge_type_ontology[rela_type]] = 1
        g.add_edge(list(node_dict[event_id])[0], ii + len(events))


    for ii in range(len(g.vs)):
        for jj in range(len(g.vs)):
            path = g.get_shortest_paths(ii, to=jj)

            length = len(path[0])
            dis_matrix[ii][jj] = length

    assert len(x_features) == len(events) + len(event_entity_relas)
    # x_features = torch.Tensor(x_features)
        
    return A_init, node_dict, g, x_features, entity_type_dict, len(events), dis_matrix, role_feature

def _construct_A_true(A_true, node_dict, g, entity_relas, entities_dict):
    for key, val in node_dict.items():
        if len(val) <= 1:
            continue
        if key not in entities_dict:
            raise
        val = list(val)
        for ii in range(len(val) - 1):
            for jj in range(ii + 1, len(val)):
                A_true[val[ii], val[jj], 1] = 0
                A_true[val[ii], val[jj], 2] = 1
                A_true[val[jj], val[ii], 1] = 0
                A_true[val[jj], val[ii], 2] = 1
                g.add_edge(val[ii], val[jj])
    seen_pairs = set()
    for ii in range(len(entity_relas)):
        start_entity, entity_rela_type, end_entity = entity_relas[ii]
        if start_entity == end_entity:
            continue
        if start_entity < end_entity:
            if start_entity + end_entity in seen_pairs:
                continue
            seen_pairs.add(start_entity + end_entity)
        else:
            if end_entity + start_entity in seen_pairs:
                continue
            seen_pairs.add(end_entity + start_entity)
        if start_entity not in set(entities_dict) or end_entity not in set(entities_dict):
            raise
        if start_entity not in node_dict or end_entity not in node_dict:
            continue
        start_ids = list(node_dict[start_entity])
        end_ids = list(node_dict[end_entity])
        
        for jj in range(len(start_ids)):
            for kk in range(len(end_ids)):
                type_id = edge_type_ontology[entity_rela_type]
                A_true[start_ids[jj], end_ids[kk], 1] = 0
                A_true[start_ids[jj], end_ids[kk], type_id] = 1
                A_true[end_ids[kk], start_ids[jj], 1] = 0
                A_true[end_ids[kk], start_ids[jj], type_id] = 1
                g.add_edge(start_ids[jj], end_ids[kk])
    return A_true, node_dict, g, entity_relas

def _convert_A_to_dense(matrix):
    ret_mat = np.zeros((len(matrix), len(matrix)))
    for ii in range(len(matrix)):
        for jj in range(len(matrix[ii])):
            if len(np.nonzero(matrix[ii][jj])[0]) != 1:
                print(np.nonzero(matrix[ii][jj]))
                raise
            ret_mat[ii, jj] = int(np.nonzero(matrix[ii][jj])[0])
    return ret_mat

# 建立图
def _construct_graph(pruned_file):
    dataset = pickle.load(open(pruned_file, "rb"))
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = filter_relations(dataset)

    all_A_init = []
    all_A_true = []
    all_dis = []
    all_x_features = []
    all_event_len = []
    all_role_f = []
    dict_dis = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0}
    for ii in range(len(all_graph_events)):
        print("iter" + str(ii))
        events = all_graph_events[ii]
        event_entity_relas = all_graph_event_entity_relations[ii]
        event_links = all_graph_event_links[ii]
        entities = all_graph_entities[ii]
        entity_relas = all_graph_entity_relations[ii]

        A_init, node_dict, g, X_features, entity_type_dict, len_event, dis_matrix, role_feature = _construct_graph_init(
            events, entities, event_entity_relas, event_links)

        A_true = np.copy(A_init)
        
        A_true, node_dict, g, entity_relas = _construct_A_true(
            A_true, node_dict, g, entity_relas, entity_type_dict)

        
        for ii in range(dis_matrix.shape[0]):
            for jj in range(dis_matrix.shape[1]):
                if ii > len_event and jj > len_event:
                    if A_true[ii,jj,1] != 1:
                        dict_dis[dis_matrix[ii][jj]]+=1

        if np.sum(A_true) != A_true.shape[0] * A_true.shape[1]:
            print("true not qeual")
            raise
        if np.sum(A_init) != A_init.shape[0] * A_init.shape[1]:
            print("init not qeual")
            raise

        A_init_dense = _convert_A_to_dense(A_init)
        A_true_dense = _convert_A_to_dense(A_true)

        all_A_init.append(A_init_dense)
        all_A_true.append(A_true_dense)
        all_x_features.append(np.array(X_features))
        all_event_len.append(len_event)
        all_dis.append(dis_matrix)
        all_role_f.append(role_feature)
    print(dict_dis)
    
    return all_A_init, all_A_true, all_x_features, all_event_len,all_dis, all_role_f

def read_dataset():
    scenarios = ["wiki_ied_bombings_", "suicide_ied_", "wiki_mass_car_bombings_"]
    

    for scenario in scenarios:
        all_test_A_init = []
        all_test_A_true = []
        all_test_x_features = []
        all_test_len = []
        all_test_dis = []
        all_test_role = []
      
        test_A_init, test_A_true, test_x_features, test_len, test_dis_matrix, test_role_f = _construct_graph(
            '../../data/Wiki_IED_split/test/' + scenario + 'test_pruned_new_no_iso_max_50_dataset_enriched.pkl')
     

        all_test_A_init += test_A_init
        all_test_A_true += test_A_true
        all_test_x_features += test_x_features
        all_test_len += test_len
        all_test_dis += test_dis_matrix
        all_test_role += test_role_f

    
        print(len(all_test_A_init))
        with open('../../data/'+scenario+'test_pruned_with_bert_max_50_set_enriched.pkl', 'wb') as handle:
            pickle.dump([all_test_A_init, all_test_A_true, all_test_x_features, all_test_len, all_test_dis,  all_test_role], handle)

read_dataset()


