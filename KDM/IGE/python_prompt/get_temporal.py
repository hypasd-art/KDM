import openai
import pickle
import ast
from igraph import *

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

def is_valid_python_code(code_string):
    try:
        ast.parse(code_string)
        return True
    except SyntaxError:
        return False

def filter_not_python_demo(code_string):
    clean_code_list = []
    code_list = code_string.split("\n")
    for code_str in code_list:
        if is_valid_python_code(code_str):
            clean_code_list.append(code_str)
    clean_code_string = "\n".join(clean_code_list)
    return clean_code_string

def get_schema_rel(instance_rel, events_dict):
    res = []
    for idx, events in enumerate(events_dict):
        s_events = []
        for event in events:
            if event.split(".")[-1] == 'Unspecified':
                s_event = event.split(".")[1]
            else:
                s_event = event.split(".")[-1]
            s_events.append(s_event)
        one_seq_rels = []
        for rel in instance_rel[idx]:
            one_seq_rel = {}
            # g = Graph(directed=True)
            name_new = None
            rel_connection = {}
            for class_tuple in rel:
                if class_tuple[0] not in s_events:
                    name_new = class_tuple[0]
                    for k, v in class_tuple[2].items():
                        list_r = []
                        for r in rel:
                            if r[0] in s_events:
                                for kr, vr in r[2].items():
                                    if vr == v:
                                        list_r.append((r[0], kr))
                        rel_connection[k] = list_r
                    break
            if name_new != None:
                one_seq_rels.append((name_new, rel_connection))
                
        res.append(one_seq_rels)
    print(res)
    return res

def write_event_rel_demo(schema_list, events_all):
    res_messgae = []
    for i, one_seq_rels in enumerate(schema_list):
        events = events_all[i]
        print(i)
        print(events)
        s_events = []
        for event in events:
            if event.split(".")[-1] == 'Unspecified':
                s_event = event.split(".")[1]
            else:
                s_event = event.split(".")[-1]
            s_events.append(s_event)
        messages = []
        for node_tuple in one_seq_rels:
            question_A = "In a complex bombing event scenario, " + ", ".join(s_events) +" happened in time order, what is the most relative choice to event "+node_tuple[0]+" in this scenario, please choose your answer from:\n"
            assert len(s_events) == 3
            answer_A = "A:" + s_events[0] + "\n" + "B:" + s_events[1] + "\n" + "C:" + s_events[2] + "\n"

            question_B = "then choose what is the most possible relation between your choice and "+ node_tuple[0] + " in a bombing event scenario, please choose your answer from:\n"
            answer_B = "A:temporal;" + node_tuple[0] + " happened first\n" + "B:temporal;" + node_tuple[0] + " happened after\n" + "C:attribute\n"

            request = "(attention you can only answer the above two questions only with A or B or C)"
            message = question_A + answer_A + question_B + answer_B + request
            messages.append(message)
        res_messgae.append(messages)
    return res_messgae

def find_instantiations(code_string):
    instantiations = []

    # Parse the code string into an Abstract Syntax Tree (AST)
    ast_tree = ast.parse(code_string)

    # Traverse the AST and find all instantiation assignments
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if isinstance(node.value.func, ast.Name):
                    # Check if the call is to a class constructor
                    class_name = node.value.func.id
                    param_names = []
                    param_values = []
                    params={}
                    
                    for keyword in node.value.keywords:
                        # Extract the value and parameter name of each keyword argument
                        arg_value = keyword.value.n if isinstance(keyword.value, ast.Num) else keyword.value.s if isinstance(keyword.value, ast.Str) else keyword.value.id if isinstance(keyword.value, ast.Name) else None
                        arg_name = keyword.arg
                        params[arg_name] = arg_value
                        param_names.append(arg_name)
                        param_values.append(arg_value)
                    try:
                        instantiations.append((class_name, node.targets[0].id, params))
                    except:
                        continue
    return instantiations

def get_rel_by_instantiations(new_classes_relation, all_events):
    result = []
    for idx, new_class_relation in enumerate(new_classes_relation):
        res_list = []
        for ids, demo in enumerate(new_class_relation):
            res = find_instantiations(demo)
            res_list.append(res)
        result.append(res_list)
    return result

def get_schema_list(name):
    new_classes_relation = []
    all_events = []
    for i in range(input_num):
        [new_classes_relation_i, all_events_i] = pickle.load(
                        open('../../../data/Wiki_IED_split/enhance_process_dir3/'+name+'_argument_rel_'+str(i)+'.pkl', 'rb'))
        new_classes_relation.append(new_classes_relation_i)

        all_events.append(all_events_i)

    for new_class_relation in new_classes_relation:
        for idx, demo in enumerate(new_class_relation):
            new_class_relation[idx] = filter_not_python_demo(demo)
            

    result = get_rel_by_instantiations(new_classes_relation, all_events)
    print(result)
    print("*"*200)
    schema_res = get_schema_rel(result, all_events)
    return schema_res, all_events

def get_final_info(name, schema_res, all_events):
    res_message = write_event_rel_demo(schema_res, all_events)
    new_classes_relation = []
    for i, demos in enumerate(res_message):
        new_class_relation = []
        for k, demo in enumerate(demos):
            print(i, k)
            text_generation = get_chatgpt_response(demo, 0.2)
            print(text_generation)
            print("\n")
            new_class_relation.append(text_generation)
        new_classes_relation.append(new_class_relation)

        print(schema_res[i])
        assert len(schema_res[i]) == len(new_classes_relation[i])
        print("*"*100)
        with open('../../../data/Wiki_IED_split/enhance_process_dir3/'+name+'_final_rel_' + str(i) + '.pkl', 'wb') as handle:
            pickle.dump([new_classes_relation[i], all_events[i], schema_res[i]], handle)

if __name__ == "__main__":
    scenario_dict = {"wiki_mass_car_bombings": "car_bombing", "wiki_ied_bombings": "bombing", "suicide_ied": "suicide bombing"}
    for name in ["suicide_ied", "wiki_ied_bombings", "wiki_mass_car_bombings"]:

        schema_res, all_events = get_schema_list(name)
        # schema_res : [
        #   (new_event, {arg1:[(old_event1, old_event1_arg),,,,,],,,,,})
        # ]

        get_final_info(name, schema_res, all_events)