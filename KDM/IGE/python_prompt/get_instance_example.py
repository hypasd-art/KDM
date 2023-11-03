import openai
import torch
import igraph
import numpy as np
import os
import pickle
import pandas as pd
import ast
from igraph import *
import random 

input_num = 10

openai.api_key = ""

def get_chatgpt_response(message, temperature = 0.6):
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


def write_rel_demo(events_list, seq_dict):
    rel_demo = []
    python_demo = []
    all_events = []
    event_dict, role_dict = _load_role_ontology()
    assert len(events_list) == len(seq_dict)
    for i, link in enumerate(seq_dict[:len(events_list)]):
        one_type_demos = []
        all_class_demo = ""
        events = []
        for event in link:
            
            if event.split(".")[-1] == 'Unspecified':
                s_event = event.split(".")[1]
            else:
                s_event = event.split(".")[-1]
            events.append(s_event)

            para = ", ".join([role[0] for role in role_dict[event]])
            class_demo = "class "+ s_event + "(event):\n" + \
                            "    def __init__(self, "+ para +"):\n"
            role_demo = ""
            for role in role_dict[event]:
                role_demo += "        self." + role[0] + "=" + role[0] + "\n"
            class_demo+= role_demo + "\n"
            all_class_demo += class_demo


        dict_events = events_list[i]
        
        for k, v in dict_events.items():
            para = ", ".join(v)
            class_demo = "class "+ k + "():\n" + \
                            "    def __init__(self, "+ para +"):\n"
            role_demo = ""
            for role in v:
                role_demo += "        self." + role + "=" + role + "\n"
            class_demo+= role_demo + "\n"
            prompt = "some python demo to model a bombing event scenario we know is above, "+ \
                "please enrich the demo by instantiating the class to explain the relation between class " + \
                ", ".join(events) + " and " + k + ". you should only generate demo to instantiate the above class\n"
            incontext_learning = 'one demo example is below:\n\
# Instantiate Attack event\n\
attack_event = Attack(attacker="John", target="Building", instrument="Explosive", place="City Center")\n\
\
# Instantiate DetonateExplode event\n\
detonate_explode_event = DetonateExplode(attacker="John", target="Building", instrument="Explosive", explosive_device="C4", place="City Center")\n\
\
# Instantiate Injure event\n\
injure_event = Injure(victim="Jane", injurer="John", instrument="Explosive")\n\
\
# Instantiate Investigate event\n\
investigate_event = Investigate(location="City Center", investigator="Detective Smith")'
            demo = all_class_demo + class_demo +prompt + incontext_learning
            # print(demo)
            one_type_demos.append(demo)
        # print("*"*200)
        # all_events.append(event)
        python_demo.append(one_type_demos)
    return python_demo

def find_classes(code_string):
    classes = {}
    try:
        ast_tree = ast.parse(code_string)
    except:
        return classes

    for node in ast.walk(ast_tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            attributes = []

            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef) and class_node.name == "__init__":
                    for sub_node in class_node.body:
                        if isinstance(sub_node, ast.Assign):
                            attribute_name = sub_node.targets[0].attr
                            attributes.append(attribute_name)
            classes[class_name] = attributes
    return classes

def select_new_class(python_demo, all_events):
    class_list = []
    freq_list = []
    for i, demos in enumerate(python_demo):
        # print(len(demos))
        events = all_events[i]
        # print(events)
        one_edge_class = {}
        freq_e = {}
        for demo in demos:
            classes = find_classes(demo.lower())
            for k, v in classes.items():
                if k in events:
                    continue
                if k in one_edge_class:
                    one_edge_class[k] = list(set(one_edge_class[k]).union(set(v)))
                else:
                    one_edge_class[k] =v 
                if k in freq_e:
                    freq_e[k] += 1
                else:
                    freq_e[k] = 1
        class_list.append(one_edge_class)
        freq_list.append(freq_e)
    return class_list, freq_list


def select_class_and_get_rel(name):
    possible_knowledge = []
    all_events = []
    for i in range(input_num):
        [possible_knowledge_i, all_events_i] = pickle.load(
                        open('../../../data/Wiki_IED_split/enhance_process_dir/'+name+'_argument_local_knowledge_dataset_'+str(i)+'.pkl', 'rb'))
        possible_knowledge.append(possible_knowledge_i)
        all_events.append(all_events_i)
    

    new_knowledge, freq_e = select_new_class(possible_knowledge, all_events)
    new_knowledge_filter = []
    print("the generated classes and their freq are:")
    for i, dict_c in enumerate(new_knowledge):
        dict_c_fliter = {}
        for k, v in dict_c.items():
            if freq_e[i][k] >= 3:
                dict_c_fliter[k] = v
                print(k, freq_e[i][k])
                print(v)
        new_knowledge_filter.append(dict_c_fliter)
        print(str(i)*100)

    all_new_demo = write_rel_demo(new_knowledge_filter, all_events)

    new_classes_relation = []
    for i, demos in enumerate(all_new_demo):
        new_class_relation = []
        for k, demo in enumerate(demos):
   
            text_generation = get_chatgpt_response(demo)


            new_class_relation.append(text_generation)
        new_classes_relation.append(new_class_relation)
        with open('../../../data/Wiki_IED_split/enhance_process_dir/'+name+'_argument_rel_' + str(i) + '.pkl', 'wb') as handle:
            pickle.dump([new_classes_relation[i], all_events[i]], handle)


if __name__ == "__main__":
    scenario_dict = {"wiki_mass_car_bombings": "car_bombing", "wiki_ied_bombings": "bombing", "suicide_ied": "suicide bombing"}
    for name in ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]:

        select_class_and_get_rel(name)