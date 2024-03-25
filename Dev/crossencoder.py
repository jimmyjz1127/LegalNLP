'''
    @author James Zhang
    @since November 1, 2023
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
from sentence_transformers import CrossEncoder
import torch 
import pickle


from ingest import *

class crossencoder():
    def __init__(self):
        self.model = CrossEncoder('distilroberta-base')
        self.documents = []
        self.embeddings = None

    def load_documents(self, path):
        '''
            Loads documents from file path 

            Arguments:
                path : path to file containing documents (json) 
        '''

        with open(path, 'r') as corpus_file:
            self.documents = json.load(corpus_file)

    def search(self, query):
        '''
            Encodes query and documents simultaneously to produce similarity scores 
        '''
        pairs = [[query, document['main']] for document in self.documents]

        scores = self.model.predict(pairs)

        for i in range(0,):
            print('{:.8f}'.format(scores[i]), self.documents[i]['name'])


def main(args):
    engine = crossencoder()

    engine.load_documents(args[1])

    engine.search(args[0])

if __name__ == '__main__':
    main(sys.argv[1:])