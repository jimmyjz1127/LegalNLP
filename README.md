# St Andrews Computer Science Senior Honours Project Code Directory: CS4099-LegalNLP
### Author : James (Jimmy) Zhang 
### Date : 22 March, 2024


## 1 Dependency Installations 
### 1.1 Python Dependencies 
1. I have included a requirements.txt file at the top level directory which can be used as follows (in any location) : pip install -r requirements.txt 
2. Alternatively, the command below can also be used (in any location) <br>
pip install -U torch transformers nltk pandas flask flask-cors faiss-cpu matplotlib scikit-learn tqdm sentence-transformers

### 1.2 Javascript Dependencies
1. Navigate to src/client
2. Execute "npm install" from terminal 



## 2 Launching the User Application
1. First install dependencies as instructed in section 1 
2. Navigate to src/server and run command : python app.py 
3. Navigate to src/client and run command : npm run start 
    - If the port is taken, run the following : PORT=\<port number\> npm start 
4. In your browser search the following URL : http://localhost:3000 or http://localhost:\<port number\>

## 3 Retrieving and using fine-tuned models 
* All fine-tuned models are located on the HuggingFace Hub and can be accesed with an internet connection 
* Within python code, the models can be retried using the from_pretrained(\<model_path\>)
Model paths : 
1. Single_TE                    : jimmyjz1127/single_te             (Textual Entalment Model)
2. Single_QA                    : jimmyjz1127/single_qa             (Selective Q&A Model)
3. Multi_Parallel               : jimmyjz1127/multi_parallel        (Parallel Multi-Task Model TE + QA)
4. Multi_Sequential             : jimmyjz1127/multi_sequential      (Sequential Multi-Task Model TE +Q A)
5. MLM + Multi_Parallel         : jimmyjz1127/multi_parallel_mlm    (Parallel Multi-Task Model with MLM Pre-training)



## 4 File Structure 
Dev/                    : Contains developmental code  
Notebooks               : All notebooks used for model fine-tuning and evaluation  
  Dev/                  : developmental notebooks for data pre-processing, cleaning, or experimenting  
  processed_data        : contains training datasets (note that the files are reduced subsets of the origina datasets)  
  evaluate.ipynb        : Notebook for evaluating the performance of the core encoder layers of the fine-tuned models  
  lm_benchmark.ipynb    : for evaluating the Witten-Bell trigram performance  
  mlm_train.ipynb       : for training the masked-language model  
  parallel_train.ipynb  : Implements the multi-task parallel training routine  
  qa_train.ipynb        : Implements the selective question and answering training routine  
  te_train.ipynb        : Implements the textual entailment training routine  
src  
  client                : Contains React.js code for implementing user interface and frontend  
  server                : Contains backend python code  
    Embeddings/          
      tfidf.pkl         : Matrix and vectorize for TF-IDF engine  
    corpus.csv          : Corpus of legal documents for search   
    app.py              : Python Flask server which implements the search piplines   
requirements.txt        : Python dependencies   
README.md  
.gitignore  
