Run the following code in order to peocess the training data and obtain the final result

python read_ontology.py 
    Store a dictionary of events, relationships, and entities, mapping types to sequence numbers
    input: ../../data/kairos_ontology.xlsx
    output: ../../data/kairos_ontology.pkl  


python get_node_type_embeddings.py
    Vector representation of storage nodes
    input: ../../data/kairos_ontology.xlsx
    output: ../../data/kairos_ontology_embeddings.pkl
        

python read_wiki.py
    Store all relationships in the corresponding dataset to facilitate further graph construction
    input: ../../data/WIki_IED_split/~/~.json
    output: ../../data/WIki_IED_split/~/~.pkl
        


    
python convert_wiki_to_ids.py
    Encode the temporal relationships of previous events, relationships between entities, and roles before events and entities into dictionary values
    input: ../../data/WIki_IED_split/~/~.pkl
    output: 
    ../../data/Wiki_IED_split/~/~_to_ids.pkl
        
    ../../data/Wiki_IED_split/~/~_igraphs.pkl
        


python prune_instance_graphs.py
    the usful information is which have the commen role in the graph

    ../../data/Wiki_IED_split/~/~_pruned_new_no_iso_max_150_igraphs.pkl
    ../../data/Wiki_IED_split/~/~_pruned_new_no_iso_max_150_dataset.pkl


python calcu_info_and_clean.py
    clean the data
    input: ../../data/Wiki_IED_split/~/~_pruned_new_no_iso_max_50_igraphs.pkl
    output: ../../data/Wiki_IED_split/~/~_pruned_new_no_iso_max_50_remove_same_type_and_unlogic_igraphs.pkl