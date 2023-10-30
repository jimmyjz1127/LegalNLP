'''
    @author James Zhang
    @since October 30, 2023
'''
import numpy as np 
import pandas as pd 
import re 
import os 
import sys
import json 
import nltk
import string

from sentence_transformers import SentenceTransformer, util
import torch 

import pickle

class model:


    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.embeddings = None
    
    def load_documents(self, path):
        '''
            Loads documents from file path 

            Arguments:
                path : path to file containing document content 
        '''

        with open(path, 'r') as corpus_file:
            self.documents = json.load(corpus_file)
    

    def generate_embeddings(self):
        '''
            Generates array of vector embeddings for entire document corpus (after loaded in)
        '''
        if len(self.documents) != 0:
            self.embeddings = self.model.encode([obj['main'] for obj in self.documents], convert_to_tensor = True)
        else :
            print('ERROR : (generate_embeddings) no documents loaded')
            sys.exit(0)


    def search(self, query):
        '''
            Performs cosine similarity between query and all documents 
        '''
        query_embedding = self.model.encode(query, convert_to_tensor = True)

        relevance_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_10 = torch.topk(relevance_scores, k=5)

        for score, idx in zip(top_10[0], top_10[1]):
            print(self.documents[idx]['name'], "(Score: {:.4f})".format(score))


    def store_embeddings(self):
        '''
            Stores embeddings into persistent file 
        '''
        if len(self.embeddings) != 0:
            with open('./Embeddings/model_embeddings.pkl', 'wb') as pickle_file:
                pickle.dump(self.embeddings, f)
        else :
            print('ERROR : (store_embeddings) There are no embeddings to store!')


    def load_embeddings(self):
        '''
            Load embeddings from pickle file 
        '''
        with open('./Embeddings/modal_embeddings.pkl', 'rb') as pickle_file:
            self.embeddings = pickle.load(pickle_file)


def main(query, filepath, flag):
    engine = model()

    engine.load_documents(filepath)

    engine.generate_embeddings()

    engine.search(query)

if __name__ == '__main__':
    '''
        Arguments
            0 : query 
            1 : path to documents 
    '''
    query, filepath, flag = None, None, 0

    if (len(sys.argv) < 3) or len(sys.argv) > 4:
        print('Invalid Arguments : python model.py <query> <path to documents corpus> [flag]')
        sys.exit(0)
    elif len(sys.argv) == 4:
        query, filepath, flag = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        query, filepath = sys.argv[1], sys.argv[2]
    
    main(query, filepath, flag)