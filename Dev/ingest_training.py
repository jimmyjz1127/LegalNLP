'''
    For ingesting training data into usable format 

    @Author James Zhang
    @Since  November 6, 2023
'''
import sys
import os 
import json 
import string 
import re 
import glob 

import matplotlib.pyplot as plt
import numpy

def r_legaladvice(path, save_path):
    '''
        For ingesting r/legaladvice training data into usable format 

        Arguments:
            path      : path to dataset file 
            save_path : path to file to save formatted data 
    '''

    data = []
    with open(path, 'r') as file:
        for line in file:
            json_line = json.loads(line)

            text = json_line['body']
            query = json_line['title']
            topic_label = json_line['text_label']

            data.append({
                "text" : text,
                "query" : query,
                "topic_label" : topic_label,
                "secondary_query" : "",
                "label" : 1
            })

    with open(save_path, 'w') as save_file:
        json.dump(data, save_file)


def aus_sum_cases(path, save_path):
    '''
        For ingesting australia cases dataset into usable format 

        Arguments:
            path      : path to dataset file 
            save_path : path to file to save formatted data 
    '''
    data = []
    with open(path, 'r') as file:
        for line in file:
            json_line = json.loads(line)

            text = json_line['text']
            query = json_line['summary']
            secondary_query = json_line['title']

            data.append({
                "text" : text,
                "query" : query,
                "topic_label" : "",
                "secondary_query" : secondary_query,
                "label" : 1
            })

    with open(save_path, 'w') as save_file:
        json.dump(data, save_file)

def eu2uk_sum_cases(judgement_path, summary_path, save_path):
    '''
        For ingesting EU to UK cases dataset into usable format 

        Arguments:
            path      : path to dataset file 
            save_path : path to file to save formatted data 
    '''
    data = []
    
    judgement_file_paths = sorted(glob.glob(os.path.join(judgement_path, '*.txt')))
    summary_file_paths = sorted(glob.glob(os.path.join(summary_path, '*.txt')))

    for judgement_filepath, summary_filepath in zip(judgement_file_paths, summary_file_paths):

        entry = {
            'secondary_query' : "",
            'topic_label' : "",
            'label' : 1,
        }

        with open(judgement_filepath, 'r', encoding='utf-8') as judgement_file:
            entry['text'] = judgement_file.read()

        with open(summary_filepath, 'r', encoding='utf-8') as summary_file:
            entry['query'] = summary_file.read()

        data.append(entry)

    with open(save_path, 'w') as save_file:
        json.dump(data, save_file)


def eu2uk_ir_legislation(data_path, corpus_path, save_path):
    '''
    
    '''
    data = []

    data_file = open(data_path, 'r')
    corpus_file = open(corpus_path, 'r')

    corpus_dict = {}

    for line in corpus_file:
        json_line = json.loads(line)
        corpus_dict[json_line['document_id']] = json_line['text']
    

    for line in data_file:

        json_line = json.loads(line)

        relevant_docs = json_line['relevant_documents']

        # Create sentence pair for each relevant document 
        for id in relevant_docs:
            data.append({
                'query' : json_line['text'],
                'text' : corpus_dict[id],
                'topic_label' : '',
                'secondary_label' : 'none',
                'label' : 1
            })
    
    with open(save_path, 'w') as save_file:
        json.dump(data, save_file)
        



def length_distribution(path, save_path):
    '''
        Generates histgram distribution for lengths of both queries and documents for formatted dataset 

        Arguments:
            path      : path to formatted data set {query, text}
            save_path : path to save distribution graph
    '''
    query_lengths = []
    doc_lengths = []

    with open(path, mode='r') as file:
        json_file = json.load(file)

        for line in json_file:

            query_lengths.append(len(line['query']))
            doc_lengths.append(len(line['text']))


    title = (path.split('/').pop()).split('.')[0]

    plt.hist(query_lengths)
    plt.title(f'{title} Query Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel(f'# of Docs (total = {len(query_lengths)})')
    plt.savefig(f'{save_path}/{title}_query_dist.png')
 
    plt.hist(doc_lengths)
    plt.title(f'{title} Document Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel(f'# of Docs (total = {len(doc_lengths)})')
    plt.savefig(f'{save_path}/{title}_doc_.png')

        

def main(args):
    flag = args[0]

    if flag == 'r_legaladvice':
        if len(args) != 3:
            print('Invalid arguments!')
            print('Usage : python ingest_training.py r_legaladvice <path to dataset file> <path to save formatted data>')
            sys.exit(0)
        else:
            r_legaladvice(args[1], args[2])
    elif flag == 'aus_sum_cases':
        if len(args) != 3:
            print('Invalid arguments!')
            print('Usage : python ingest_training.py aus_sum_cases <path to dataset file> <path to save formatted data>')
            sys.exit(0)
        else:
            aus_sum_cases(args[1], args[2])
    elif flag == 'eu2uk_sum_cases':
        if len(args) != 4:
            print('Invalid arguments!')
            print('Usage : python ingest_training.py eu2uk_sum_cases <path/to/judgement/files> <path/to/summary/files> <path/to/save/formatted/data>')
            sys.exit(0)
        else:
            eu2uk_sum_cases(args[1], args[2], args[3])
    elif flag == 'eu2uk_ir_legislation':
        if len(args) != 4:
            print("Invalid Arguments!")
            print('Usage : python ingest_training.py eu2uk_ir_legislation <path/to/data/file> <path/to/corpus/file> <path to file to save graph>')
        else:
            eu2uk_ir_legislation(args[1], args[2], args[3])
    elif flag == 'distribution':
        if len(args) != 3:
            print("Invalid Arguments!")
            print('Usage : python ingest_training.py distribution <path to data file> <path to file to save graph>')
        else : 
            length_distribution(args[1], args[2])


if __name__ == '__main__':
    main(sys.argv[1:])