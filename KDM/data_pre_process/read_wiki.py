"""
The code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import json
import numpy as np
import pickle

"""
Read different datasets separately and store the data in each instance graph in a pkl file, including all events, 
event roles, event sequences, entities, and entity relationships

all_events : (eventname, evnet-type)
all_event_entity_relation: event_name, relation_type, end_entity_name
all_event_links: (start_event_name, end_event_name)
all_entities : (entity_name, entity_type)
all_entity_relations :  (start_entityname, entity_rela_type, end_entity_name)
"""
def read_dataset(file_name):
    all_graph_events = []
    all_graph_event_entity_relations = []
    all_graph_event_links = []
    all_graph_entities = []
    all_graph_entity_relations = []
    for line in open(file_name + '.json', 'r'):
        line = line.rstrip('\n')
        graph_obj = json.loads(line)
        instance_graph = graph_obj['schemas'][0]
        all_events, all_event_entity_relations = collect_events(instance_graph['steps'])
        all_event_links = collect_event_links(instance_graph['order'])
        all_entities = collect_entities(instance_graph['entities'])
        all_entity_relations = collect_entity_relations(instance_graph['entityRelations'])
            
        all_graph_events.append(all_events)
        all_graph_event_entity_relations.append(all_event_entity_relations)
        all_graph_event_links.append(all_event_links)
        all_graph_entities.append(all_entities)
        all_graph_entity_relations.append(all_entity_relations)

    with open(file_name + '.pkl', 'wb') as handle:
        pickle.dump([all_graph_events, all_graph_event_entity_relations,
            all_graph_event_links, all_graph_entities,
            all_graph_entity_relations], handle)


def collect_entities(entities):
    all_entities = []
    for ii in range(len(entities)):
        entity_name = entities[ii]['@id']
        entity_type = entities[ii]['entityTypes'].split('/')[-1]
        all_entities.append((entity_name, entity_type))
    return all_entities

def _relation_naming(relation_type_str):
    relation_type_core = relation_type_str.split('/')[-1]
    if len(relation_type_core.split('.')) == 2:
        if relation_type_core.endswith('Resident'):
            relation_type_str = relation_type_str + '.Resident'
        elif relation_type_core.endswith('OrganizationHeadquarters'):
            relation_type_str = relation_type_str + '.OrganizationHeadquarters'
        elif relation_type_core.endswith('Founder'):
            relation_type_str = relation_type_str + '.Founder'
        elif relation_type_core.endswith('Make'):
            relation_type_str = relation_type_str + '.Make'
        elif relation_type_core.endswith('Color'):
            relation_type_str = relation_type_str + '.Color'
        elif relation_type_core.endswith('StudentAlum'):
            return None
        else:
            relation_type_str = relation_type_str + '.Unspecified'
    return relation_type_str

def collect_entity_relations(entity_relations):
    all_entity_relations = []
    for ii in range(len(entity_relations)):
        start_entity = entity_relations[ii]['relationSubject']
        curr_relation = entity_relations[ii]['relations']
        end_entity = curr_relation['relationObject']
        entity_rela_type = curr_relation['relationPredicate'].split('/')[-1]
        entity_rela_type = _relation_naming(entity_rela_type)
        all_entity_relations.append((start_entity, entity_rela_type, end_entity))
    return all_entity_relations

def collect_events(events):
    all_events = []
    all_event_entity_relations = []
    for ii in range(len(events)):
        event_name = events[ii]['@id']
        event_type = events[ii]['@type'].split('/')[-1]
        all_events.append((event_name, event_type))
        event_entity_relations = events[ii]['participants']
        for jj in range(len(event_entity_relations)):
            relation_type = event_entity_relations[jj]['role'].split('/')[-1]
            for kk in range(len(event_entity_relations[jj]['values'])):
                end_entity = (event_entity_relations[jj]['values'][kk]['entity'])
                all_event_entity_relations.append((event_name, relation_type, end_entity))
    return all_events, all_event_entity_relations

def collect_event_links(events_orders):
    all_event_links = []
    for ii in range(len(events_orders)):
        start_event = events_orders[ii]['before']
        end_event = events_orders[ii]['after']
        all_event_links.append((start_event, end_event))
    return all_event_links




file_dir = "../../data/Wiki_IED_split/"
dataset_types = ["train", "dev", "test"]
file_names = ["suicide_ied_", "wiki_ied_bombings_", "wiki_mass_car_bombings_"]


for file_name in file_names:
    for dataset_type in dataset_types:
        curr_file = file_dir + dataset_type + "/" + file_name + dataset_type
        read_dataset(curr_file)








