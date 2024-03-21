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

from pprint import pprint

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
        self.corpus = df
        self.breakdown_documents(df)

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

    def breakdown_documents(self, documents):
        ''' 
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
        keys = ['name', 'date','court','attorneys','extra']

        new_documents = [] 

        for index, document in documents.iterrows():

            if len(document['main']) > 1024:
                sections = self.breakdown_document(document['main'])

                for section in sections:
                    obj = {} 
                    obj['main'] = section
                    for key in keys:
                        obj[key] = document[key]
                    new_documents.append(obj)
            else:
                new_documents.append(document)

        self.documents = pd.DataFrame(new_documents)
                    


    def breakdown_document(self, document, max_length=1024, stride = 128):
        def find_split_index(s, start):
            end = min(start + max_length, len(s))

            if end == len(s): return len(s)

            split_index = s.rfind(' ', start, end)
            return split_index if split_index != -1 else end

        sections = []
        start = 0
        while start < len(document):
            split_index = find_split_index(document, start)
            sections.append(document[start:split_index].strip())
            # start = split_index + 1 if split_index < len(paragraph) else len(paragraph)

            if start + stride >= len(document) or split_index >= len(document): break

            next_start = document.rfind(' ', start, start+stride)

            start = next_start + 1 if next_start != -1 else len(document)

        return sections
    
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

        # Generate the query vector
        query_vector = self.vectorizer.transform([query])
        # Calculate cosine similarities between the query and the corpus
        similarities = linear_kernel(query_vector, self.corpus_tfidf).flatten()
        
        # Initialize an empty list for storing unique ranked documents
        unique_ranked_documents = []
        # Keep track of names that have been added
        added_names = set()
        
        # Iterate over documents and their similarities
        for i, score in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True):
            if score > 0:
                doc = self.documents.loc[i].copy()
                # Check if the document's name is unique
                if doc['name'] not in added_names:
                    name = doc['name']
                    og_main = self.corpus.loc[self.corpus['name'] == name, 'main'].iloc[0]
                    
                    # Since 'doc' is a copy for reading, you can modify it before appending
                    doc['main'] = og_main
                    unique_ranked_documents.append((doc, score))
                    added_names.add(doc['name'])
            
            # Break the loop if we've collected enough unique results
            if len(unique_ranked_documents) == k:
                break
                
        return unique_ranked_documents


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

        # self.model = transformers.BertModel.from_pretrained(model_path, output_attentions=True)
        self.model = transformers.BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('casehold/legalbert')

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.model.to(self.device)

    def bi_search(self, query):
        top_k_tfidf = self.tf_idf.search(query, 10)

        df_rows = [row for row,_ in top_k_tfidf]

        dataframe = pd.concat(df_rows, axis=1).transpose()

        query_tokens = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        query_tokens = {key: value.to(self.device) for key, value in query_tokens.items()}
        
        with torch.no_grad():
            # query_embedding = self.model(**query_tokens).last_hidden_state.mean(dim=1)
            query_output = self.model(**query_tokens)
            query_embedding = query_output.last_hidden_state.mean(dim=1)
            query_attention = query_output.attentions

        similarity_scores = []

        for main_text in dataframe['main']:

            # Tokenize and encode the text for the model input
            text_tokens = self.tokenizer(main_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            text_tokens = {key: value.to(self.device) for key, value in text_tokens.items()}
            
            # Get text embedding
            with torch.no_grad():
                # text_embedding = self.model(**text_tokens).last_hidden_state.mean(dim=1)
                main_output = self.model(**text_tokens)
                text_embedding = main_output.last_hidden_state.mean(dim=1)
            
            # Compute cosine similarity and append to list
            similarity = cosine_similarity(query_embedding.cpu().numpy(), text_embedding.cpu().numpy())[0][0]
            similarity_scores.append(similarity)

        query_attention = [t.tolist() for t in query_attention]

        # Add similarity scores to the dataframe
        dataframe['similarity'] = similarity_scores 
        
        # Sort the dataframe by similarity scores in descending order
        sorted_dataframe = dataframe.sort_values(by='similarity', ascending=False)
        
        return sorted_dataframe


    def cross_search(self, query):
        top_k_tfidf = self.tf_idf.search(query, 10)
        df_rows = [row for row, _ in top_k_tfidf]
        dataframe = pd.concat(df_rows, axis=1).transpose()
        
        # Initialize a list to hold the attention data for each query-document pair
        query_attention_list = []
        similarity_scores = []
        query_tokens = []

        for main_text in dataframe['main']:
            # Combine the query and main_text into a single sequence
            combined_input = query + " [SEP] " + main_text
            
            # Tokenize combined input
            tokens = self.tokenizer(combined_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            
            with torch.no_grad():
                # Ensure the model outputs attention weights
                model_output = self.model(**tokens, output_attentions=True)
                attention_weights = model_output.attentions  # This will be a tuple of tensors

                logits = model_output.logits
                score = torch.nn.functional.softmax(logits, dim=1)[:,1].item()
                similarity_scores.append(score)
           
            q_tokens = self.tokenizer.tokenize(query)
            query_tokens.append(json.dumps(q_tokens))
            # Determine the range of token indices for the query
            query_token_length = len(q_tokens)
            # Adjust indices for any special tokens (e.g., [CLS] at the start)
            start_index = 1  # Assuming [CLS] is the first token
            end_index = start_index + query_token_length
            
            # Extract attention weights for the query tokens from the last layer 
            query_attention = attention_weights[-1][0, :, start_index:end_index, start_index:end_index].mean(dim=0)
            query_attention_list.append(query_attention.tolist())
            
        # Add similarity scores to the dataframe
        dataframe['similarity'] = similarity_scores 
        dataframe['attention'] = query_attention_list
        dataframe['attention'] = dataframe['attention'].apply(lambda x: json.dumps(x))
        dataframe['query_tokens'] = query_tokens
        
        # Sort the dataframe by similarity scores in descending order
        sorted_dataframe = dataframe.sort_values(by='similarity', ascending=False)

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

            results= self.engine.cross_search(data['query']).to_json(orient='records')

    
            return jsonify(isError=False, message="Success", statusCode=200, data=results), 200

    def run(self, debug=True):
        self.app.run(debug=debug)



if __name__ == '__main__':
    server = FlaskServer()
    server.run()