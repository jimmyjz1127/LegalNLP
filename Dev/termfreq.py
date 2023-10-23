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

def ingest(data_path, schema_path):
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
            if count > 15:return corpus

            json_line = json.loads(line)

            entry = {}

            for key, val in schema.items():
                obj = json_line
                for elem in val:
                    obj = obj[f'{elem}']
                entry[key] = obj

            entry['id'] = entry['name'].translate(str.maketrans('', '', string.punctuation)).replace(' ', '')
        
            corpus.append(entry)

    return corpus



def main(filepath, schemapath, query, flag):
    data = ingest(filepath, schemapath)

    engine = tfidf_corp()

    engine.add_documents(data)

    engine.generate_tfidf()

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

    main(filepath, schemapath, query, flag)