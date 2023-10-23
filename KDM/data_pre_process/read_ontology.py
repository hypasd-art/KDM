"""
The code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import pandas as pd
import pickle
import json
import numpy as np

"""
read RESIN_schema, build the dictionary about the type
event : 67   ArtifactExistence.DamageDestroyDisableDismantle.Damage
event_entity_relation(role): 85   PaymentBarter
entity : 24  ABS
entity_relation : 46  Evaluate.Belief.CommittedBelief
"""
def read_ontology(file_path):
    events_ontology = pd.read_excel(file_path, sheet_name="events")
    entities_ontology = pd.read_excel(file_path, sheet_name="entities")
    relations_ontology = pd.read_excel(file_path, sheet_name="relations")
    event_type_dict = construct_events_types(events_ontology)
    # print(event_type_dict) # 67
    event_entity_relation_dict = construct_event_entity_relations(events_ontology)
    # print(event_entity_relation_dict) # 85
    entity_types_dict = construct_entities_types(entities_ontology)
    # print(entity_types_dict)# 24
    entity_relation_dict = construct_entity_relations_types(relations_ontology)
    # print(entity_relation_dict) # 46

    saved_dict = {"event_types": event_type_dict,
        "event_entity_relations": event_entity_relation_dict,
        "entity_types_dict": entity_types_dict,
        "entity_relation_dict": entity_relation_dict}

    with open('../../data/kairos_ontology.pkl', 'wb') as handle:
        pickle.dump([saved_dict], handle)


def construct_events_types(ontology):
    event_types = ontology['Type'].tolist()
    event_subtypes = ontology['Subtype'].tolist()
    event_sub_subtypes = ontology['Sub-subtype'].tolist()
    event_type_dict = {}
    for ii in range(len(event_types)):
        event_type_dict[event_types[ii] + "." + event_subtypes[ii] +
            '.' + event_sub_subtypes[ii]] = ii
    return event_type_dict


def construct_event_entity_relations(ontology):
    event_entity_relations = (ontology['arg1 label'].tolist() +
        ontology['arg2 label'].tolist() + ontology['arg3 label'].tolist() +
        ontology['arg4 label'].tolist() + ontology['arg5 label'].tolist() +
        ontology['arg6 label'].tolist())
    event_entity_relations = set(event_entity_relations)
    event_entity_relations.remove(np.nan)
    event_entity_relations = list(event_entity_relations)
    event_entity_relation_dict = {}
    for ii in range(len(event_entity_relations)):
        event_entity_relation_dict[event_entity_relations[ii]] = ii
    return event_entity_relation_dict


def construct_entities_types(ontology):
    entity_types = ontology['Type'].tolist()
    entity_type_dict = {}
    for ii in range(len(entity_types)):
        entity_type_dict[entity_types[ii]] = ii
    return entity_type_dict


def construct_entity_relations_types(ontology):
    entity_types = ontology['Type'].tolist()
    entity_subtypes = ontology['Subtype'].tolist()
    entity_sub_subtypes = ontology['Sub-subtype'].tolist()
    entity_relation_dict = {}
    for ii in range(len(entity_types)):
        entity_relation_dict[entity_types[ii] + "." + entity_subtypes[ii] +
            '.' + entity_sub_subtypes[ii]] = ii
    return entity_relation_dict


read_ontology("../../data/kairos-ontology.xlsx")



















