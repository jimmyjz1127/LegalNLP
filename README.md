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




