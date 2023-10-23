import pickle
from igraph import *
import copy
import random
max_node = 50
input_num = 10

def _load_ontology_dict_all():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types = saved_dict["event_types"]
    event_entity_relations = saved_dict["event_entity_relations"]
    entity_types_dict = saved_dict["entity_types_dict"]
    entity_relation_dict = saved_dict["entity_relation_dict"]

    node_type_dict = {}
    for key, val in event_types.items():
        node_type_dict[key] = val
    for key, val in entity_types_dict.items():
        node_type_dict[key] = val + 67

    edge_type_list = ["event-event-before-after","no-edge","identical-edge"]
    for key, val in event_entity_relations.items():
        edge_type_list.append(key)
    for key, val in entity_relation_dict.items():
        edge_type_list.append(key)

    return node_type_dict, edge_type_list

def _load_event_ontology():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {"START": 0, "END": 1}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[key] = val + 2
    return event_types_ontology_new

def _load_event_ontology_re():
    saved_dict = pickle.load(open("../../../data/kairos_ontology.pkl", "rb"))[0]
    event_types_ontology = saved_dict['event_types']
    event_types_ontology_new = {0:"START", 1:"END"}
    for key, val in event_types_ontology.items():
        event_types_ontology_new[val + 2] = key
    return event_types_ontology_new

def _construct_event_name_to_id_dict(all_graph_events, events_ontology):
    event_dict = {}
    for ii in range(len(all_graph_events)):
        for jj in range(len(all_graph_events[ii])):
            curr_entity = all_graph_events[ii][jj]
            event_dict[curr_entity[0]] = events_ontology[curr_entity[1]]
    return event_dict

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
        # print(event_entity_shared_dict)
        
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
                    # print(start, end)
                    # print(val)
                    valid_event_links.append((start, end))

        # print(len(valid_event_links), len(curr_graph_event_links))
        valid_event_links = _remove_same_type_links(valid_event_links,event_name_to_id)

        all_valid_event_links.append(valid_event_links)
    return all_valid_event_links

def inject_entity_to_data(
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_events, 
    event_name_to_id,
    all_graph_entity_relations,
    all_graph_entities,
    name
    ):
    [new_event_dict, new_event_con] = pickle.load(open("../../../data/ontology_new_con.pkl", "rb"))

    all_graph_events_en = copy.deepcopy(all_graph_events)
    all_graph_event_entity_relations_en = copy.deepcopy(all_graph_event_entity_relations)
    all_graph_event_links_en = copy.deepcopy(all_graph_event_links)
    all_graph_entity_relations_en = copy.deepcopy(all_graph_entity_relations)
    all_graph_entities_en = copy.deepcopy(all_graph_entities)

    dict_event_name_to_id = _load_event_ontology_re()
    node_type_dict, edge_type_list = _load_ontology_dict_all()

    new_classes_relation = []
    all_events = []
    schema_res = []
    for i in range(input_num):
        new_classes_relation_i_clean = []
        [new_classes_relation_i, all_events_i, schema_res_i] = pickle.load(
                        open('../../../data/Wiki_IED_split/'+name+'_final_rel_'+str(i)+'.pkl', 'rb'))
        assert len(new_classes_relation_i) == len(schema_res_i)
        for i, rel in enumerate(new_classes_relation_i):
            answer = rel.split("\n")
            answer_clean = []
            while "" in answer:
                answer.remove("")
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
    
    for ii in range(len(all_graph_event_entity_relations)):
        inject_max = 10
        inject = 0
        event = all_graph_events_en[ii]
        event_entity = all_graph_event_entity_relations_en[ii]
        event_link = all_graph_event_links_en[ii]
        entity_relation = all_graph_entity_relations_en[ii]
        entity = all_graph_entities_en[ii]
        # print(len(event), len(event_entity), len(event_link), len(entity), len(entity_relation))

        curr_graph_event_entity_links = all_graph_event_entity_relations[ii]
        curr_graph_event_links = all_graph_event_links[ii]

        for jj in range(len(curr_graph_event_links)):
            start_j, end_j = curr_graph_event_links[jj]
            for kk in range(len(curr_graph_event_links)):
                start_k, end_k = curr_graph_event_links[kk]
                if end_j == start_k and inject < inject_max:
                    complex_id = [start_j, end_j, end_k]
                    type_seq = (dict_event_name_to_id[event_name_to_id[start_j]], dict_event_name_to_id[event_name_to_id[end_j]], dict_event_name_to_id[event_name_to_id[end_k]])
                    # print(type_seq)
                    for ind, schema in enumerate(all_events):
                        classes_rel_init = new_classes_relation[ind] 
                        entity_schema_res = schema_res[ind]
                        classes_rel = []
                        entity_rel = []
                        for idx, t in enumerate(classes_rel_init):
                            if t[2].lower() in new_event_dict:
                                classes_rel.append(t)
                                entity_rel.append(entity_schema_res[idx])
                        if type_seq == schema and random.random() < 0.6:
                            if len(classes_rel) == 0:
                                continue
                            inject += 1
                            if inject > inject_max:
                                break
                            num = random.randint(0,len(classes_rel)-1)
                            rel = classes_rel[num]
                            entity_to_add = entity_rel[num]

                            added_type = dict_event_name_to_id[new_event_dict[rel[2].lower()]]

                            event.append(("event:new:"+str(jj)+str(kk), added_type))

                            index = rel[0][1]
                            event_complex_id = complex_id[index]
                            if rel[1] == "tB":
                                event_link.append(("event:new:"+str(jj)+str(kk), event_complex_id))
                            elif rel[1] == "tA":
                                event_link.append((event_complex_id, "event:new:"+str(jj)+str(kk)))
                            
                            
                            for k,v in entity_to_add[1].items():
                                if len(v) > 0:
                                    label_rel = v[0][1]
                                    for e_e in curr_graph_event_entity_links:
                                        if e_e[0] == event_complex_id and label_rel == e_e[1]:
                                            # print(e_e[2])
                                        # print(("event:new:"+str(jj)+str(kk), k ,"entity"+str(jj)+str(kk)+k))
                                        # entity.append(("entity"+str(jj)+str(kk)+k, "PER"))
                                            if k.capitalize() in edge_type_list:
                                                event_entity.append(("event:new:"+str(jj)+str(kk), k.capitalize(), e_e[2]))
                                                # print("yyy")
                                        # entity_relation.append(("event:new:"+str(jj)+str(kk)))
        # print(len(event), len(event_entity), len(event_link), len(entity), len(entity_relation))
        
        all_graph_events_en[ii] = event
        all_graph_event_entity_relations_en[ii] = event_entity
        all_graph_event_links_en[ii] = event_link
        all_graph_entity_relations_en[ii] = entity_relation
        all_graph_entities_en[ii] = entity
    return all_graph_events_en, all_graph_event_entity_relations_en, all_graph_event_links_en, all_graph_entity_relations_en, all_graph_entities_en

def construct_event_igraphs(file_name, plt_name, name):
    [all_graph_events, 
    all_graph_event_entity_relations,
    all_graph_event_links, 
    all_graph_entities,
    all_graph_entity_relations] = pickle.load(open(file_name + "_pruned_new_no_iso_max_50_dataset.pkl", "rb"))
    
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
    intx = 0
    for ind in range(len(all_graph_events)):
        if len(all_graph_events[ind]) > 50:
            continue
        intx += 1
    old_all_graph_events = copy.deepcopy(all_graph_events)
    all_graph_events, all_graph_event_entity_relations, all_valid_event_links,  all_graph_entity_relations, all_graph_entities = inject_entity_to_data(
        all_graph_event_entity_relations, 
        all_valid_event_links, 
        all_graph_events,
        event_name_to_id, 
        all_graph_entity_relations,
        all_graph_entities,
        name)

    
    all_updated_graph_events = []
    all_updated_graph_event_entity_relations = []
    all_updated_graph_event_links = []
    all_updated_graph_entities = []
    all_updated_graph_entity_relations = []
    inty = 0
    for ind in range(len(all_graph_events)):
        inty += 1
        all_updated_graph_events.append(all_graph_events[ind])
        all_updated_graph_event_entity_relations.append(all_graph_event_entity_relations[ind])
        all_updated_graph_event_links.append(all_valid_event_links[ind])
        all_updated_graph_entities.append(all_graph_entities[ind])
        all_updated_graph_entity_relations.append(all_graph_entity_relations[ind])
    print(intx, inty)


    with open(file_name + '_pruned_new_no_iso_max_'+ str(max_node) + '_dataset_enriched.pkl', 'wb') as handle:
        pickle.dump([
            all_updated_graph_events, 
            all_updated_graph_event_entity_relations,
            all_updated_graph_event_links, 
            all_updated_graph_entities,
            all_updated_graph_entity_relations], handle)
    
if __name__ == "__main__":
    file_dir = "../../../data/Wiki_IED_split/"
    dataset_types = ["train", "dev", "test"]
    file_names = ["suicide_ied_", "wiki_ied_bombings_", "wiki_mass_car_bombings_"]
    
    for file_name in file_names:
        for dataset_type in dataset_types:
            curr_file = file_dir + dataset_type + "/" + file_name + dataset_type
            construct_event_igraphs(curr_file, file_name + dataset_type, file_name[:-1])
