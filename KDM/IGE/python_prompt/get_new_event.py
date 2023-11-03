import pickle
import openai
import igraph
import numpy as np
import os
import pickle
import pandas as pd
import ast

input_num = 10
openai.api_key = ""

def get_chatgpt_response(message, temperature = 0.8):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages = [
        {"role":"system","content":"you have the strong ability to write detailed storys and you can understand the logic in complex events about suicide bombing"},
        {"role":"user", "content":message}
    ],
    temperature = temperature
    )
    return response["choices"][0]["message"]["content"]

def is_nan(nan):
    return nan != nan
    
def _load_event_ontology():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {0: "START", 1: "END"}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[val + 2] = key

    return event_types_ontology_new

def _load_role_ontology(file_path = "../../../data/kairos-ontology.xlsx"):
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
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
        key = event_types[ii] + "." + event_subtypes[ii] + '.' + event_sub_subtypes[ii]
        event_entity_rel_dict[key] = role_list
    assert len(event_types_ontology_new) == len(event_entity_rel_dict) + 2
    return event_types_ontology_new, event_entity_rel_dict


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


def write_python_demo(name, seq_dict, topk = 10):
    python_demo = []
    all_events = []
    event_dict, role_dict = _load_role_ontology()# event_dict:id->complete_name    role_dict:{name:[[arg, con], [arg,con],,,]}
    for link in seq_dict[:topk]:
        all_class_demo = ""
        events = []
        for event in link[0]:
            events.append(event)
            if event.split(".")[-1] == 'Unspecified':
                s_event = event.split(".")[1]
            else:
                s_event = event.split(".")[-1]
            para = ", ".join([role[0] for role in role_dict[event]])
            class_demo = "class "+ s_event + "(event):\n" + \
                            "    def __init__(self, "+ para +"):\n"
            role_demo = ""
            for role in role_dict[event]:
                role_demo += "        self." + role[0] + "=" + role[0] + "\n"
            class_demo+= role_demo + "\n"
            all_class_demo += class_demo
        prompt = "some python demo to model a bombing event scenario we know is above, please enrich the demo with new 'named entity' and 'event' python class. you should not repeat the python demo above!you should not repeat the python demo above!you should not repeat the python demo above!"
        demo = all_class_demo + prompt
        all_events.append(events)
        python_demo.append(demo)
    return python_demo, all_events
  
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

def get_all_graph_edge(g_list):
    # id->complete name
    event_types_ontology_new = _load_event_ontology()
    dict_seq2 = {}
    dict_seq3 = {}
    for [graph, depth] in g_list:
        res_dict2, res_dict3 = get_seq(graph, event_types_ontology_new)
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
    dict_seq2 = sorted(dict_seq2.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    dict_seq3 = sorted(dict_seq3.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    # print(dict_seq2)
    # print("*"*100)
    # print(dict_seq3)
    return dict_seq2, dict_seq3


def get_gpt_class_responce(name):

    g_list = load_data(name)

    dict_2, dict_3 = get_all_graph_edge(g_list)

    python_demo, all_events = write_python_demo(name, dict_3, topk=10)


    possible_knowledge = [[] for i in range(len(python_demo))]

    for i, demo in enumerate(python_demo):
        for k in range(input_num):
            print(f"generating {i}, {k}")
            text_generation = get_chatgpt_response(demo, 0.8)
            print(text_generation)
            print("\n"*3)
            possible_knowledge[i].append(text_generation)
        print("saved in " + '../../../data/Wiki_IED_split/enhance_process_dir/' + name + '_argument_local_knowledge_dataset_' + str(i) + '.pkl')
        with open('../../../data/Wiki_IED_split/enhance_process_dir/' + name + '_argument_local_knowledge_dataset_' + str(i) + '.pkl', 'wb') as handle:
            pickle.dump([possible_knowledge[i], all_events[i]], handle)

if __name__ == "__main__":
    scenario_dict = {"wiki_mass_car_bombings": "car_bombing", "wiki_ied_bombings": "bombing", "suicide_ied": "suicide bombing"}
    for name in ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]:

        get_gpt_class_responce(name)