import os
import sys
import json
import re
import string
import random
import time
import datetime

import numpy as np
import pandas as pd


import transformers
from transformers import BertTokenizer, BertModel, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS  # Import CORS
from flask_cors import cross_origin
import time

import pickle
nltk.download('stopwords')


class tfidf_corp:
    '''
        Class definition of tfidf_corp object for building TF-IDF matrix of document corpus and performing 
        cosine similarity searches.
    '''


    def __init__(self, datapath):
        '''
            Constructor : initializes vectorizer object, corpus TF-IDF matrix, empty document list, and stopword list
        '''
        self.vectorizer = TfidfVectorizer()
        self.corpus_tfidf = None
        self.documents = []
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))
        self.datapath = datapath

    def set_documents(self, df):
        self.documents = df

    def load_documents(self):
        with open(self.datapath, 'r') as corpus_file:
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

        self.corpus_tfidf = self.vectorizer.fit_transform([obj['main'] for idx,obj in self.documents.iterrows()])

    def search(self, query, k):
        '''
            Performs cosine similarity search for query against document corpus 
        '''

        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.corpus_tfidf).flatten()

        ranked_documents = [(self.documents.loc[i], score) for i, score in enumerate(similarities) if score > 0]
        ranked_documents.sort(key=lambda x: x[1], reverse=True)

        return ranked_documents[0:k]


    def store_matrix(self, path):
        '''
            Saves TF-IDF matrix into pickle file 
        '''

        with open(path, 'wb') as pickle_file:
            pickle.dump((self.vectorizer, self.corpus_tfidf), pickle_file)


    def load_matrix(self, path):
        ''' 
            Loads TF-IDF matrix from pickle file 
        '''

        with open('Embeddings/tfidf.pkl', 'rb') as pickle_file:
            self.vectorizer, self.corpus_tfidf = pickle.load(pickle_file) # need to save both vectorizer object and matrix to file


class SearchEngine():
    def __init__(self, model_path, corpus_path):
        self.df = pd.read_json(corpus_path)

        self.tf_idf = tfidf_corp(corpus_path)
        self.tf_idf.set_documents(self.df)
        self.tf_idf.generate_tfidf()

        self.model = transformers.BertModel.from_pretrained(model_path)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('casehold/legalbert')

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.model.to(self.device)

    def search(self, query):
        top_k_tfidf = self.tf_idf.search(query, 10)

        df_rows = [row for row,_ in top_k_tfidf]

        dataframe = pd.concat(df_rows, axis=1).transpose()

        query_tokens = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        query_tokens = {key: value.to(self.device) for key, value in query_tokens.items()}
        
        with torch.no_grad():
            query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)

        similarity_scores = []

        for main_text in dataframe['name']:

            # Tokenize and encode the text for the model input
            text_tokens = self.tokenizer(main_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            text_tokens = {key: value.to(self.device) for key, value in text_tokens.items()}
            
            # Get text embedding
            with torch.no_grad():
                text_embedding = self.model(**text_tokens).last_hidden_state.mean(dim=1)
            
            # Compute cosine similarity and append to list
            similarity = cosine_similarity(query_embedding.cpu().numpy(), text_embedding.cpu().numpy())[0][0]
            similarity_scores.append(similarity)

        # Add similarity scores to the dataframe
        dataframe['similarity'] = similarity_scores
        
        # Sort the dataframe by similarity scores in descending order
        sorted_dataframe = dataframe.sort_values(by='similarity', ascending=False)
        
        # Optionally, you might want to drop the similarity column before returning
        # sorted_dataframe.drop(columns=['similarity'], inplace=True)
        
        return sorted_dataframe
    

class FlaskServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, supports_credentials=True, origins="http://localhost:3000", allow_headers=["Content-Type"])
        self.setup_routes()
        self.engine = SearchEngine('./../../Notebooks/models/mlm_model_manual', './../../Dev/corpus.json')

    def setup_routes(self):
        @self.app.route('/', methods=['GET'])
        def home():
            return "Welcome to the Flask server!"

        @self.app.route('/submit', methods=['POST'])
        def submit():
            data = request.json
            response = {
                'status': 'success',
                'data_received': data
            }
            return jsonify(response)

        @self.app.route('/search', methods=['POST'])
        def search():
            data = request.json

            results = self.engine.search(data['query']).to_json(orient='records')

    
            return jsonify(isError=False, message="Success", statusCode=200, data=results), 200

    def run(self, debug=True):
        self.app.run(debug=debug)



if __name__ == '__main__':
    server = FlaskServer()
    server.run()