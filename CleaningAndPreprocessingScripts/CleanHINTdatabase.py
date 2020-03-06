#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hesham El Abd
@brief: Cleaning the HINT Database. 
"""
## loading the modules: 
import os 
import pandas as pd
import pickle 
## get the folder_name:
org_dirs=[direc for direc in os.listdir() if "." not in direc]
## define a pairs list:
pair_lst=set()
## iterating and getting the data names: 
for org_dir in org_dirs:
    print("I am loading data from the directory :"+org_dir+"/n")
    files=os.listdir(org_dir+"/")
    for file in files: 
        print("file:"+org_dir+"/n")
        dum_table=pd.read_table(org_dir+"/"+file)
        for a_tuple in dum_table.itertuples():
            pair_lst.add((a_tuple[1],a_tuple[2]))
        print("current database size is : {}".format(len(pair_lst)))
## make a directory to store the results: 
os.mkdir('results')
## save the results as a pickle object on the results directory 
with open('results/set_pairs.pickle','wb') as binary_writer:
    pickle.dump(pair_lst,binary_writer)
## getting the set of unique proteins
unique_proteins=set()
for protein_pair in pair_lst:
    unique_proteins.add(protein_pair[0])
    unique_proteins.add(protein_pair[1])
print("number of unique proteins is {}".format(len(unique_proteins)))
## save the results as a pickle object on the results directory 
with open('results/unique_protein_set.pickle', 'wb') as binary_writer:
    pickle.dump(unique_proteins,binary_writer)
