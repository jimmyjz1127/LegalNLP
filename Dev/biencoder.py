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
from sentence_transformers import SentenceTransformer, util, InputExample
from torch.utils.data import DataLoader
import torch 
import pickle


from ingest import *

class biencoder:


    def __init__(self):
        # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
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
            return (len(self.documents))
        else :
            print('ERROR : (generate_embeddings) no documents loaded')
            sys.exit(0)

    def append_embeddings(self, corpus_path, append_path, embedding_path):
        '''
            Generates embeddings for one or more new documents and appends documents and embeddings to respective files 

            Arguments:
                corpus_path     : path to document corpus (json)
                append_path     : path to file containing documents to append (json)
                embedding_path  : path to file containing corpus embeddings (pickle)
        '''

        # Retrieve ids from current corpus 
        with open(corpus_path, 'r') as corpus_file:
            corpus = json.load(corpus_file)
            ids = set([obj['id'] for obj in corpus])

        with open(append_path, 'r') as doc_file:
            new_documents = json.load(doc_file)
            docs_to_append = []

            # Filter our documents that already exist in corpus (based on id)
            for document in new_documents:
                if document['id'] not in ids: docs_to_append.append(document)

            # Append new documents to corpus file
            ingest.update_corpus(docs_to_append)

            # Calculate and append new embeddings 
            self.embeddings = torch.cat([self.embeddings, self.model.encode(docs_to_append, convert_to_tensor=True)], dim=0)

            # Store new set of embeddings in pickle file
            self.store_embeddings(embedding_path)

            return len(docs_to_append)

    def search(self, query):
        '''
            Performs cosine similarity between query and all documents 
        '''
        query_embedding = self.model.encode(query, show_progress_bar=True, convert_to_tensor = True)

        relevance_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_10 = torch.topk(relevance_scores, k=5)

        for score, idx in zip(top_10[0], top_10[1]):
            print("{:.8f}".format(score), self.documents[idx]['name'])


    def store_embeddings(self, path):
        '''
            Stores embeddings into persistent file 
        '''
        if len(self.embeddings) != 0:
            with open(path, 'wb') as pickle_file:
                pickle.dump(self.embeddings, pickle_file)
        else :
            print('ERROR : (store_embeddings) There are no embeddings to store!')


    def load_embeddings(self, path):
        '''
            Load embeddings from pickle file 
        '''
        with open(path, 'rb') as pickle_file:
            self.embeddings = pickle.load(pickle_file)

    def train(self, path):
        '''
            For training model with a given dataset 

            Arguments:
                path : path to file containing training data 
        '''


 

def generate_and_store_embeddings(corpus_path, embedding_path):
    '''
        Generates embeddings for a corpus of documents and stores the embeddings in a specified file 

        Arguments
            corpus_path     : path to file containing document corpus (json)
            embeddinng_path : path to file to store embeddings in (pickle)
    '''
    print('Generating embeddings...')

    engine = biencoder()

    engine.load_documents(corpus_path)

    num_docs = engine.generate_embeddings()

    engine.store_embeddings(embedding_path)

    print(f'Complete! \n Generated Embeddings for {num_docs} documents')

def generate_and_append_embeddings(corpus_path, append_path, embedding_path):
    '''
        Generates embeddings for one or more new documents and appends documents and embeddings to respective files 

        Arguments:
            corpus_path     : path to document corpus (json)
            append_path     : path to file containing documents to append (json)
            embedding_path  : path to file containing corpus embeddings (pickle)
    '''

    print('Generating embeddings...')

    engine = biencoder()

    engine.load_embeddings(embedding_path)

    num_docs = engine.append_embeddings(corpus_path, append_path, embedding_path)

    print(f'Complete! \n Generated and appended embeddings for {num_docs} documents.')

def query(query, embedding_path, corpus_path):
    '''
        Executes query over corpus to retrieve results 

        Arguments:
            query       : the query to search (string)
            corpus_path : path to document corpus file (json)
    '''

    engine = biencoder()

    engine.load_documents(corpus_path)

    engine.load_embeddings(embedding_path)

    engine.search(query)


def main(args):
    if len(args) < 1:
        print('Insufficient arguments!')
        sys.exit(0)

    flag = args[0]

    if flag == '-q':
        if len(args) != 4:
            print('Invalid usage : -q <query> <path/to/embeddings/file> <path/to/corpus/file>')
            sys.exit(0)
        else:
            query(args[1], args[2], args[3])
    elif flag == '-a':
        if len(args) != 4:
            print('Invalid usage : -a <path/to/corpus/file> <path/to/documents/to/append> <path/to/embeddings/file>')
            sys.exit(0)
        else:
            generate_and_append_embeddings(args[1], args[2], args[3])
    elif flag == '-w':
        if len(args) != 3:
            print('Invalid usage : -w <path/to/corpus/file> <path/to/embedding/file>')
            sys.exit(0)
        else:
            generate_and_store_embeddings(args[1], args[2])


if __name__ == '__main__':
    main(sys.argv[1:])