import faiss 
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS  # Import CORS
from flask_cors import cross_origin
import time


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



class FaissIndex:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)  # Flat index for inner product similarity
        
    def add_documents(self, embeddings):
        self.index.add(embeddings)
    
    def search(self, query_embedding, k):
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    

class SearchEngine:
    def __init__(self, model_path, corpus_path, flag):
        self.corpus_df = pd.read_csv(corpus_path)
        self.model = transformers.BertModel.from_pretrained(model_path)
        self.tokenizer = transformers.BertTokenizer.from_pretrained('casehold/legalbert')

        self.device = 'cpu'
        if torch.cuda.is_available() : self.device = 'cuda'
        self.model.to(self.device)

        self.faiss_index = FaissIndex(embedding_dim=768)  # Assuming BERT produces 768-dimensional embeddings
        
        # Preprocess and add documents to the FAISS index
        self.add_documents_to_index()
    
    def add_documents_to_index(self):
        corpus_embeddings = np.array(self.generate_embeddings(self.corpus_df['main'])['Embeddings'].tolist())
        self.faiss_index.add_documents(corpus_embeddings)
    
    def generate_embeddings(self, documents):
        embeddings = []
        attention_list = []

        for document in documents:
            if  type(document) is not type("String"): continue
            tokenized_text = self.tokenizer(document, return_tensors='pt', padding=True, truncation=True,  max_length=512)
            # tokens = {key: value.to(self.device) for key, value in tokenized_text.items()}
            with torch.no_grad():
                model_output = self.model(**tokenized_text, output_attentions=True)
                document_embedding = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(document_embedding)

                attention_weights = model_output.attentions  # This will be a tuple of tensors


            tokens = self.tokenizer.tokenize(document)

            # Determine top 10 document tokens based on attention weight 
            start_index_m = 1  # Assuming [SEP] token between query and document
            end_index_m = start_index_m + len(tokens)

            document_attention = attention_weights[-1][0, :, start_index_m:end_index_m, start_index_m:end_index_m].mean(dim=0)
            k = min(10, len(document_attention))
            
            top_attentions, top_indices = torch.topk(document_attention, k)  # Get top 10 attentions and their indices
            # Ensure top_indices and top_attentions are properly flattened
            top_indices_flat = top_indices.cpu().numpy().flatten()
            top_attentions_flat = top_attentions.cpu().numpy().flatten()

            # Assuming top_indices_flat and top_attentions_flat are defined as before
            top_tokens_with_weights = {tokens[idx]: float(attention) for idx, attention in zip(top_indices_flat, top_attentions_flat)}
            
            attention_list.append(json.dumps(top_tokens_with_weights))

        df = pd.DataFrame({'Embeddings':embeddings, "Attentions":attention_list})

        # return np.array(embeddings)
        return df
    
    def search(self, query, k):
        # Generate embedding for the query
        query_embedding = np.array(self.generate_embeddings([query])['Embeddings'].tolist())[0]
        # Use FAISS to perform semantic search
        distances, indices = self.faiss_index.search(query_embedding.reshape(1, -1), k)
        # Return the top-k documents from the corpus
        top_k_documents = self.corpus_df.iloc[indices[0]].to_dict(orient='records')
        return top_k_documents
    

class FlaskServer:
    def __init__(self, flag):
        self.app = Flask(__name__)
        CORS(self.app, supports_credentials=True, origins="http://localhost:3000", allow_headers=["Content-Type"])
        self.setup_routes()
        self.engine = SearchEngine('./../../Notebooks/models/mlm_model_manual1', 'corpus.csv', flag)

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
    args = sys.argv[1:]

    flag = args[0]


    server = FlaskServer(flag)
    server.run()