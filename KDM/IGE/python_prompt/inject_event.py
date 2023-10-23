import igraph
import pickle
from igraph import *
import random 

input_num = 10

def load_data(name, t="train", max_n_data = 50):
    
    g_list_train = []
    max_n = 0  # maximum number of nodes

    [graphs_train, depth] = pickle.load(
        open('../../../data/Wiki_IED_split/'+t+'/' + name + '_'+ t + '_pruned_new_no_iso_max_' + str(int(max_n_data)) + '_remove_same_type_and_unlogic_igraphs.pkl', 'rb'))
    assert len(graphs_train) == len(depth)
    graphs_train = [graphs_train]
    
    for mm in range(len(graphs_train)):
        for i, row in enumerate(graphs_train[mm]):
            g, n = row
            max_n = max(max_n, n)
            g_list_train.append([g, depth[i]])

    return g_list_train

def get_node_depth(graphs):
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
            path = g.get_all_simple_paths(start, to=i)
            d = int(mean([len(path[i]) for i in range(len(path))]))
            if d > max_d:
                max_d = d 
            graph_depth.append(d)
        path_depth.append(graph_depth)
    print(f"depth: {max_d}")
    return path_depth

def _load_event_ontology():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {0: "START", 1: "END"}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[val + 2] = key

    return event_types_ontology_new

def calcu_new_classes_info():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    link_event_dict = {}
    for key, val in event_types_ontology.items():
        if key.split(".")[-1] == 'Unspecified':
            key = key.split(".")[1]
        else:
            key = key.split(".")[-1]
        link_event_dict[key.lower()] = val
    
    
    link_event_dict = {'damage': 0, 
                       'destroy': 1, 
                       'disabledefuse': 2, 
                       'dismantle': 3, 
                        'damagedestroydisabledismantle': 4, 
                        'assemble': 5, 
                        'identifycategorize': 6, 
                        'sensoryobserve': 7, 
                        'research': 8, 
                        'teachingtraininglearning': 9, 
                        'detonateexplode': 10, 
                        'attack_no': 11, 
                        'defeat': 12, 
                        'demonstratewithviolence': 13, 
                        'demonstrate': 14, 
                        'broadcast': 27, 
                        'correspondence': 28, 
                        'meet': 29, 
                        'contact': 18, 
                        'prevarication': 22, 
                        'requestcommand': 26, 
                        'threatencoerce': 30, 
                        'impedeinterferewith': 31, 
                        'crash': 32, 
                        'diseaseoutbreak': 33, 
                        'fireexplosion': 34, 
                        'genericcrime': 35, 
                        'acquit': 36, 
                        'arrest': 37, 
                        'chargeindict': 38, 
                        'convict': 39, 
                        'investigate': 40, 
                        'releaseparole': 41, 
                        'sentence': 42, 
                        'interrogate': 43, 
                        'consume': 44, 
                        'die': 45, 
                        'illness': 46, 
                        'infect': 47, 
                        'injure_no': 48, 
                        'diagnosis': 49, 
                        'rescue': 50, 
                        'vaccinate': 51, 
                        'evacuate': 52, 
                        'grantallowpassage': 53, 
                        'illegaltransportation': 54, 
                        'preventpassage': 55, 
                        'transportation': 56, 
                        'changejoblocation': 57, 
                        'demotion': 58, 
                        'lateral': 59, 
                        'promotion': 60, 
                        'changeposition': 61, 
                        'endposition': 62, 
                        'startposition': 63, 
                        'aidbetweengovernments': 64, 'donation': 65, 
                        'purchase': 66
                        }

    new_event_dict = {}
    for name in ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]:
        for i in range(input_num):
            [new_classes_relation_i, all_events_i, schema_res_i] = pickle.load(
                            open('../../../data/Wiki_IED_split/enhance_process_dir3/'+name+'_final_rel_'+str(i)+'.pkl', 'rb'))
            for event_tuple in schema_res_i:
                if event_tuple[0].lower() not in new_event_dict:
                    if event_tuple[0].lower() in link_event_dict:
                        new_event_dict[event_tuple[0].lower()] = link_event_dict[event_tuple[0].lower()]+2
                    else:
                        continue
    print(new_event_dict)
    return new_event_dict

def inject_knowledge_to_graphs(name, new_event_dict, td = 'train'):
    g_list = load_data(name,t=td)

    new_classes_relation = []
    all_events = []
    schema_res = []
    for i in range(input_num):
        new_classes_relation_i_clean = []
        [new_classes_relation_i, all_events_i, schema_res_i] = pickle.load(
                        open('../../../data/Wiki_IED_split/enhance_process_dir3/'+name+'_final_rel_'+str(i)+'.pkl', 'rb'))
        assert len(new_classes_relation_i) == len(schema_res_i)
        for i, rel in enumerate(new_classes_relation_i):
            
            answer = rel.split("\n")
            
            answer_clean = []
            while "" in answer:
                answer.remove("")
            if len(answer) != 2:
                continue
            assert len(answer) == 2
            if "A:" in answer[0]:
                answer_clean.append((all_events_i[0],0))
            elif "B:" in answer[0]:
                answer_clean.append((all_events_i[1],1))
            elif "C:" in answer[0]:
                answer_clean.append((all_events_i[2],2))
            
            if "A:" in answer[1]:
                answer_clean.append("tB")
            elif "B:" in answer[1]:
                answer_clean.append("tA")
            elif "C:" in answer[1]:
                answer_clean.append("attr")
            answer_clean.append(schema_res_i[i][0])

            if len(answer_clean) != 3:
                continue
            else:
                new_classes_relation_i_clean.append(answer_clean)
        new_classes_relation.append(new_classes_relation_i_clean)
        all_events.append(tuple(all_events_i))
        schema_res.append(schema_res_i)

    g_list = search_seq_and_inject(g_list, all_events, new_classes_relation, new_event_dict, p = 0.6)
    graph = []
    for [g, d] in g_list:
            for node in g.vs:
                assert node['type'] < 69
                
    for [g, d] in g_list:
        while True:
            judge = False
            for i in range(len(g.vs)):
                node = g.vs[i]
                if node['type'] == 0 or node['type'] == 1:
                    g.delete_vertices(node)
                    judge = True
                    break
            if judge == False:
                break
        indegree_0, outdegree_0 = _get_nodes_for_start_end(g, len(g.vs))
        num = len(g.vs)
        g.add_vertices(2)
        g.vs[num]['type'] = 0
        g.vs[num+1]["type"] = 1

        for ii in range(0, num):
            if ii in indegree_0:
                g.add_edge(num, ii)
            if ii in outdegree_0:
                g.add_edge(ii, num+1)
        g = _reorder_graph(g)
        graph.append((g, len(list(g.vs))))
    depth  = get_node_depth(graph)
    with open('../../../data/Wiki_IED_split/'+td+'/' + name + '_'+td+'_pruned_new_no_iso_max_50_remove_same_type_knowledge_inject_p_6_igraphs_version_9.pkl', 'wb') as handle:
        pickle.dump([graph, depth], handle)

def _get_nodes_for_start_end(graph, node_num):
    indegree_0 = []
    outdegree_0 = []
    for ii in range(0, node_num):
        if graph.vs[ii].indegree() == 0:
            indegree_0.append(ii)
        if graph.vs[ii].outdegree() == 0:
            outdegree_0.append(ii)
    return set(indegree_0), set(outdegree_0)

def search_seq_and_inject(g_list, schema_olds, classes_rels, new_event_dict, p, max_add_node = 10):
    dict_event = _load_event_ontology()

    inject_cal = {k:0 for k in new_event_dict.keys()}
    # print(inject_cal)
    for [res, depth] in g_list:
        inject_cal_p = {k:0 for k in new_event_dict.keys()}
        res_seq_set2 = []
        res_seq_set3 = []
        res_set3 = {}
        num_g = len(res.vs)
        max_add_node = int(num_g/6)
        for i in res.get_edgelist():
            if res.vs[i[0]]['type'] == 0 or res.vs[i[1]]['type'] == 1:
                continue
            res_seq_set2.append(i)
    
        for e in list(res_seq_set2):
            for i in list(res_seq_set2):
                if e[1] == i[0]:
                    seq = (e[0],e[1],i[1])
                    res_seq_set3.append(seq)
        add_num = 0
        for e in res_seq_set3:
            
            for ind, schema_old in enumerate(schema_olds):
                temp = False
                classes_rel_init = classes_rels[ind] 
                classes_rel = []
                for t in classes_rel_init:
                    if t[2].lower() in new_event_dict:
                        classes_rel.append(t)
                if len(classes_rel) == 0:
                    continue
                r = (dict_event[res.vs[e[0]]['type']], dict_event[res.vs[e[1]]['type']], dict_event[res.vs[e[2]]['type']])
                if r == schema_old and random.random() < p:
                    if add_num > max_add_node:
                        continue
                    num = random.randint(0,len(classes_rel)-1)
                    rel = classes_rel[num]
                    if inject_cal[rel[2].lower()] >= 70:
                        continue
                    if inject_cal_p[rel[2].lower()] >= 1:
                        continue
                    add_num += 1
                    
                    inject_cal[rel[2].lower()]+=1
                    inject_cal_p[rel[2].lower()]+=1
                    
                    node = res.add_vertex(type=new_event_dict[rel[2].lower()])
                    # print(node)
                    assert new_event_dict[rel[2].lower()] < 69
                    index = rel[0][1]
                    
                    node_old = res.vs[e[index]]
                    
                    if rel[1] == "tB":
                        res.add_edge(node, node_old)
                    elif rel[1] == "tA":
                        res.add_edge(node_old, node)
        
        assert num_g == len(res.vs) - add_num
    print(inject_cal)
    return g_list
       
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


if __name__ == "__main__":
    scenario_dict = {"wiki_mass_car_bombings": "car_bombing", "wiki_ied_bombings": "bombing", "suicide_ied": "suicide bombing"}
    for name in ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]:

        new_event_dict = calcu_new_classes_info()


        inject_knowledge_to_graphs(name, new_event_dict)