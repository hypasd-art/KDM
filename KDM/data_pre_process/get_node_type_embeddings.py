"""
Part of code is copied and adapted from https://aclanthology.org/2022.naacl-main.147/
"""
import pandas as pd
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import argparse
import pickle
import numpy as np 

file_path = "../../data/kairos-ontology.xlsx"

events_ontology = pd.read_excel(file_path, sheet_name="events")
entities_ontology = pd.read_excel(file_path, sheet_name="entities")
relations_ontology = pd.read_excel(file_path, sheet_name="relations")


event_types_def = events_ontology['Definition'].tolist()
entity_types_def = entities_ontology['Definition'].tolist()
relation_type_def = relations_ontology['Definition'].tolist()

node_types_dict = {}
for ii in range(len(event_types_def)):
    node_types_dict[ii] = event_types_def[ii]
for ii in range(len(entity_types_def)):
    node_types_dict[ii + 67] = entity_types_def[ii]

device = torch.device("cuda:6")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

model.to(device)

event_types_str_lst = [node_types_dict[ii] for ii in range(len(node_types_dict))]

event_type_tensors = tokenizer(event_types_str_lst, padding=True, truncation=True, return_tensors="pt")
# print(event_type_tensors)
event_type_embeddings = None

for i, j in enumerate(tqdm(event_types_str_lst)):
    output = model(input_ids=event_type_tensors["input_ids"][i:i+1].to(device), 
        attention_mask=event_type_tensors['attention_mask'][i:i+1].to(device))['pooler_output']
    # print(output.shape) (1,768)
    curr_out = np.array(output.squeeze().tolist()).reshape((1, -1))

    if event_type_embeddings is None:
        event_type_embeddings = curr_out
    else:
        event_type_embeddings = np.concatenate((event_type_embeddings, curr_out))

# print(event_type_embeddings.shape) # 67 + 24


with open("../../data/kairos_ontology_embeddings.pkl", "wb") as f:
    pickle.dump(event_type_embeddings, f)




















