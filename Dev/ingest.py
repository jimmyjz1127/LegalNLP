import numpy as np 
import pandas as pd 
import re 
import os 
import sys
import json 
import string


def ingest(data_path, schema_path, mode):
    '''
        Ingests documents based on schmea file (rule based parsing)

        {
            name            : name of document 
            main            : main text of document 
            date            : date of document 
            jurisdiction    : jurisdiction of document 
            judges          : judges of document 
            court           : court assocaited with document 
            attorneys       : attorneys associated with document 
            extra           : any extra text involved such as opinions or summaries 
        }
    '''

    corpus = []

    # parse schema file 
    with open(schema_path, mode='r') as schema_file:
        schema = json.load(schema_file)

    # parse data file 
    with open(data_path, mode='r') as file:
        count = 0
        for line in file:
            count += 1
            if count > 500:break

            json_line = json.loads(line)

            entry = {}

            for key, val in schema.items():
                obj = json_line
                for elem in val:
                    obj = obj[f'{elem}']
                # if not obj['name'] or not obj['main']:continue # if missing name or main field, skip
                entry[key] = obj

            entry['id'] = entry['name'].translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
            corpus.append(entry)

    # Save new data to files 
    if mode == 'a': update_corpus(corpus)
    elif mode == 'w': store_corpus(corpus)


def store_corpus(data):
    '''
        WRITES json corpus data to json file 

        Arguments : 
            data : array of json objects [ {key:val} ]
    '''
    with open('corpus.json', 'w') as corpus_file:
        json.dump(data, corpus_file)


def update_corpus(data):
    '''
        Appends new data to existing corpus (if not already included)

        Arguments : 
            data : array of json objects [ {key:val} ]
    '''

    with open('corpus.json', 'r') as corpus_file:
        corpus = json.load(corpus_file)
    
    corpus_ids = [obj['id'] for obj in corpus]

    for obj in data:
        if obj['id'] in corpus_ids: continue
        else: corpus.append(obj)

    with open(corpus.json, 'w') as corpus_file:
        json.dump(corpus, corpus_file)


def main(data_path, schema_path, mode):
    ingest(data_path, schema_path, mode)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Invalid Arguments : python ingest.py [path to data file] [path to parsing schema file] [storage mode (w or a)]')
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])