'''
    @author : James Zhang 
    @since  : October 4, 2023
'''


import numpy as np 
import pandas as pd 
import re 
import os 
import sys
import json 
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

import pickle

class tfidf_corp:
    '''
        Class definition of tfidf_corp object for building TF-IDF matrix of document corpus and performing 
        cosine similarity searches.
    
        Document Format
            {
                name : String
                date : String 
                body : String 
            }
    '''


    def __init__(self):
        '''
            Constructor : initializes vectorizer object, corpus TF-IDF matrix, empty document list, and stopword list
        '''
        self.vectorizer = TfidfVectorizer()
        self.corpus_tfidf = None
        self.documents = []
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))

    def load_documents(self):
        with open('corpus.json', 'r') as corpus_file:
            self.documents = json.load(corpus_file)

    def add_document(self, document):
        '''
            Appends a single document objects to documents list class-attribute 
        '''
        self.documents.append(document)

    def add_documents(self, documents):
        '''
            Appends list of documents to documents list class-attribute 
        '''
        self.documents = self.documents + documents
    
    def generate_tfidf(self):
        '''
            Computes TF-IDF matrix for document corpus 
        '''

        if len(self.documents) < 1:
            print('No documents in corpus')
            return

        self.corpus_tfidf = self.vectorizer.fit_transform([obj['main'] for obj in self.documents])

    def search(self, query):
        '''
            Performs cosine similarity search for query against document corpus 
        '''

        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.corpus_tfidf).flatten()

        ranked_documents = [(self.documents[i]['name'], score) for i, score in enumerate(similarities) if score > 0]
        ranked_documents.sort(key=lambda x: x[1], reverse=True)

        return ranked_documents


    def store_data(self):
        '''
            Saves TF-IDF matrix into pickle file 
        '''

        with open('tfidf.pkl', 'wb') as pickle_file:
            pickle.dump((self.vectorizer, self.corpus_tfidf), pickle_file)


    def load_data(self):
        ''' 
            Loads TF-IDF matrix from pickle file 
        '''

        with open('tfidf.pkl', 'rb') as pickle_file:
            self.vectorizer, self.corpus_tfidf = pickle.load(pickle_file)


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
            if count > 15:break

            json_line = json.loads(line)

            entry = {}

            for key, val in schema.items():
                obj = json_line
                for elem in val:
                    obj = obj[f'{elem}']
                entry[key] = obj

            entry['id'] = entry['name'].translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
        
            corpus.append(entry)

    # Save new data to files 
    if mode == 'a': update_corpus(corpus)
    elif mode == 'w': store_corpus(corpus)

    # return corpus

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


def main1(filepath, schemapath, query, flag):
    ingest(filepath, schemapath, 'w')

    engine = tfidf_corp()

    engine.load_documents()

    engine.generate_tfidf()

    engine.store_data()

    ranked_documents = engine.search(query if query else 'illinois defendent')

    for doc, score in ranked_documents:
            print(f"Document: {doc}")
            print(f"Cosine Similarity Score: {score:.4f}\n")


def main2(filepath, schemapath, query, flag):
    print('test a')
    ingest(filepath, schemapath, 'w')

    print('test b')
    engine = tfidf_corp()

    print('test c')
    engine.load_documents()

    print('test d')
    engine.load_data() 

    ranked_documents = engine.search(query if query else 'illinois defendent')

    for doc, score in ranked_documents:
            print(f"Document: {doc}")
            print(f"Cosine Similarity Score: {score:.4f}\n")


if __name__ == "__main__":
    filepath, schemapath, query, flag = None, None, None, None

    if len(sys.argv) == 5:
        filepath, schemapath, query, flag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    elif len(sys.argv) == 4:
        filepath, schemapath, query = sys.argv[1], sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 3 :
        filepath, schemapath = sys.argv[1], sys.argv[2]
    else:
        print('Invalid arguments : python termfreq.py [data filepath] [schema filepath] [query] [flag]')
        sys.exit()

    main2(filepath, schemapath, query, flag)