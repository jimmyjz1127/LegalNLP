import numpy as np 
import pandas as pd 
import re 
import os 
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
    
    '''

    '''
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

        self.corpus_tfidf = self.vectorizer.fit_transform([obj['body'] for obj in self.documents])

    def search(self, query):
        '''
            Performs cosine similarity search for query against document corpus 
        '''

        query_vector = self.vectorizer.transform([query])
        similarities = linear_kernel(query_vector, self.corpus_tfidf).flatten()

        ranked_documents = [(self.documents[i]['name'], score) for i, score in enumerate(similarities)]
        ranked_documents.sort(key=lambda x: x[1], reverse=True)

        for doc, score in ranked_documents:
            print(f"Document: {doc}")
            print(f"Cosine Similarity Score: {score:.4f}\n")



def jsonl_file_to_tokens(filepath):
    tokenized_corpus = []

    with open(filepath, mode='r') as file:
        count = 0 
        for line in file:

            count += 1
            if count == 10:
                return tokenized_corpus
            
            json_line = json.loads(line)

            url = json_line['url']
            name = json_line['name']
            short_name = json_line['name_abbreviation']
            date = json_line['decision_date']
            docket = json_line['docket_number']
            first_page = json_line['first_page']
            last_page = json_line['last_page']
            citations = json_line['citations']
            volume = json_line['volume']
            court = json_line['court'] 
            jurisdiction = json_line['jurisdiction'] 

            body = json_line['casebody']['data']['head_matter']
            opinions = json_line['casebody']['data']['opinions'] # text, author, type
            judges = json_line['casebody']['data']['judges'] 


            # body_tokens = pipeline_process(body)
            # opinions_tokens = (pipeline_process(opinions['text']) + pipeline_process(opinions['author']))

            tokenized_corpus.append({
                'url':url,
                'long_name':name,
                'name':short_name,
                'date' : date,
                'docket' : docket,
                'first_page' : first_page,
                'last_page' : last_page,
                'citations' : citations,
                'volume':volume,
                'court' : court,
                'jurisdiction' : jurisdiction,
                'body' : body,
                'opinions' : opinions,
                'judges' : judges,
            })

    return tokenized_corpus
    


def main():
    data = jsonl_file_to_tokens('./data/data.jsonl')

    engine = tfidf_corp()

    engine.add_documents(data)

    engine.generate_tfidf()

    engine.search('Ann M. Osborn and Thomas Osborn, Plaintiffs in Error, v. Jacob Horine')




if __name__ == "__main__":
    main()