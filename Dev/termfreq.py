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

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

import pickle

class tfidf_corp:
    '''
        Class definition of tfidf_corp object for building TF-IDF matrix of document corpus and performing 
        cosine similarity searches.
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

            Arguments:
                document : document json object (main, name, ..., extra)
        '''
        self.documents.append(document)

    def add_documents(self, documents):
        '''
            Appends list of documents to documents list class-attribute 

            Arguments:
                documents : list of document json objects [{main, name, ..., extra}]
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


    def store_matrix(self):
        '''
            Saves TF-IDF matrix into pickle file 
        '''

        with open('Embeddings/tfidf.pkl', 'wb') as pickle_file:
            pickle.dump((self.vectorizer, self.corpus_tfidf), pickle_file)


    def load_matrix(self):
        ''' 
            Loads TF-IDF matrix from pickle file 
        '''

        with open('Embeddings/tfidf.pkl', 'rb') as pickle_file:
            self.vectorizer, self.corpus_tfidf = pickle.load(pickle_file) # need to save both vectorizer object and matrix to file


def create_store_matrix_main(engine):
    '''
        Creates and stores TF-IDF matrix (to be run only after documents have been loaded into engine object)

        Arguments:
            engine : tfidf_corpus object
    '''

    # generate TF-IDF matrix for documents loaded 
    engine.generate_tfidf()

    # Store the TF-IDF matrix into pickle file 
    engine.store_matrix()


def main(query, flag):
    '''
        Arguments:
            query : the query to run against corpus 
            flag  : 1 to compute TF-IDF matrix, 0 otherwise
    '''

    engine = tfidf_corp()

    # load documents from corpus file 
    engine.load_documents()

    if flag : create_store_matrix_main(engine) 

    # Load matrix from pickle file
    engine.load_matrix() 

    ranked_documents = engine.search(query if query else 'illinois defendent')

    for doc, score in ranked_documents:
            print(f"Document: {doc}")
            print(f"Cosine Similarity Score: {score:.4f}\n")


if __name__ == "__main__":
    query, flag = None, 0

    if (len(sys.argv) > 3):
        print('Invalid Usage : python termfreq.py [query <optional>] [flag <optional>]')
    elif (len(sys.argv) == 3): query,flag = sys.argv[1], sys.argv[2]
    elif (len(sys.argv) == 2): query = sys.argv[1]
    
    main(query, flag)

