import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,  ChatPromptTemplate
import csv
from typing import Dict, List, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import gradio as gr
import torch
from transformers import pipeline
from llama_index import VectorStoreIndex, download_loader
from langchain.document_loaders import (
    PyPDFLoader,
    DataFrameLoader,
    GitLoader
  )
import pandas as pd
import nbformat
from nbconvert import PythonExporter
import os
import fireworks.client
from config.dev import DevConfig
from langchain_community.document_loaders import WikipediaLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

csv_file_path = os.path.join(BASE_DIR,"docs/maneno.csv")

DB_FAISS_PATH = "vectorstore/db_faiss/"

def get_text_splits(text_file):
  """Function takes in the text data and returns the  
  splits so for further processing can be done."""
  with open(text_file,'r') as txt:
    data = txt.read()

  textSplit = RecursiveCharacterTextSplitter(chunk_size=150,
                                             chunk_overlap=15,
                                             length_function=len)
  doc_list = textSplit.split_text(data)
  return doc_list
     


def get_pdf_splits(pdf_file):
  """Function takes in the pdf data and returns the  
  splits so for further processing can be done."""
  
  loader = PyPDFLoader(pdf_file)
  pages = loader.load_and_split()  

  textSplit = RecursiveCharacterTextSplitter(chunk_size=150,
                                             chunk_overlap=15,
                                             length_function=len)
  doc_list = []
  #Pages will be list of pages, so need to modify the loop
  for pg in pages:
    pg_splits = textSplit.split_text(pg.page_content)
    doc_list.extend(pg_splits)

  return doc_list

def get_excel_splits(excel_file,target_col,sheet_name):
  trialDF = pd.read_excel(io=excel_file,
                          engine='openpyxl',
                          sheet_name=sheet_name)
  
  df_loader = DataFrameLoader(trialDF,
                              page_content_column=target_col)
  
  excel_docs = df_loader.load()

  return excel_docs


def get_csv_splits(csv_file):
  """Function takes in the csv and returns the  
  splits so for further processing can be done."""
  csvLoader = CSVLoader(csv_file)
  csvdocs = csvLoader.load()
  return csvdocs

def get_ipynb_splits(notebook):
  """Function takes the notebook file,reads the file 
  data as python script, then splits script data directly"""

  with open(notebook) as fh:
    nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

  exporter = PythonExporter()
  source, meta = exporter.from_notebook_node(nb)

  #Python file data is in the source variable
  
  textSplit = RecursiveCharacterTextSplitter(chunk_size=150,
                                             chunk_overlap=15,
                                             length_function=len)
  doc_list = textSplit.split_text(source)
  return doc_list 



def get_git_files(repo_link, folder_path, file_ext):
  # eg. loading only python files
  git_loader = GitLoader(clone_url=repo_link,
    repo_path=folder_path, 
    file_filter=lambda file_path: file_path.endswith(file_ext))
  #Will take each file individual document
  git_docs = git_loader.load()

  textSplit = RecursiveCharacterTextSplitter(chunk_size=150,
                                             chunk_overlap=15,
                                             length_function=len)
  doc_list = []
  #Pages will be list of pages, so need to modify the loop
  for code in git_docs:
    code_splits = textSplit.split_text(code.page_content)
    doc_list.extend(code_splits)

  return doc_list

def get_website_splits(urls_list):
   
   loader = WebBaseLoader(urls_list)
   docs = loader.load()
   textSplit = RecursiveCharacterTextSplitter(chunk_size=150,
                                             chunk_overlap=15,
                                             length_function=len)
   doc_list = []
   for web in docs:
      web_splits = textSplit.split_text(web.page_content)
      doc_list.extend(web_splits)

   return doc_list

def embed_index(doc_list, embed_fn, index_store):
  """Function takes in existing vector_store, 
  new doc_list and embedding function that is 
  initialized on appropriate model. Local or online. 
  New embedding is merged with the existing index. If no 
  index given a new one is created"""
  #check whether the doc_list is documents, or text
  try:
    faiss_db = FAISS.from_documents(doc_list, 
                              embed_fn)  
  except Exception as e:
    faiss_db = FAISS.from_texts(doc_list, 
                              embed_fn)
  
  if os.path.exists(index_store):
    local_db = FAISS.load_local(index_store,embed_fn)
    #merging the new embedding with the existing index store
    local_db.merge_from(faiss_db)
    print("Merge completed")
    local_db.save_local(index_store)
    print("Updated index saved")
  else:
    faiss_db.save_local(folder_path=index_store)
    print("New store created...")



def embed_fn(model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def rag_local_docs():

    csv_docs = get_csv_splits(csv_file_path)

    embeddings = embed_fn()

    embed_index(doc_list=csv_docs,
                embed_fn=embeddings,
                index_store=DB_FAISS_PATH)
    


    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings)

    retriever=docsearch.as_retriever(search_kwargs={"k": 20})

    llm = CTransformers(model=os.path.join(BASE_DIR,"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)

    # Define the system message template
    system_template = """The provided {context} is a tabular dataset containing a english words and their meanings.
    The dataset includes the following columns:
    'id': unique id of the English word,
    'title': English word ,
    "meaning": the meaning of the English word in English, make sure you respond using English language to each query.
    ----------------
    {context}"""

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False,combine_docs_chain_kwargs={"prompt": qa_prompt},memory=memory,verbose=True)

    return qa


def rag_local_web():

    csv_docs = get_website_splits(urls_list=['https://www.nsoma.me'])

    embeddings = embed_fn()

    embed_index(doc_list=csv_docs,
                embed_fn=embeddings,
                index_store=DB_FAISS_PATH)
    


    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings)

    retriever=docsearch.as_retriever(search_kwargs={"k": 20})

    llm = CTransformers(model=os.path.join(BASE_DIR,"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.1)

    # Define the system message template
    system_template = """The provided {context} is a website content make sure you 
    retrieve relevant information and stay true, 
    accurate and provide links of the webpages you got the most revelant information
    {context}"""

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False,combine_docs_chain_kwargs={"prompt": qa_prompt},memory=memory,verbose=True)

    return qa

def wikipedia(prompt):
    
    wiki_docs = WikipediaLoader(query=prompt, load_max_docs=3).load()

    embeddings = embed_fn()

    embed_index(doc_list=wiki_docs,
                embed_fn=embeddings,
                index_store=DB_FAISS_PATH)
    


    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings)

    retriever=docsearch.as_retriever(search_kwargs={"k": 20})

    llm = CTransformers(model=os.path.join(BASE_DIR,"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
                        model_type="llama",
                        max_new_tokens=512,
                        temperature=0.7)

    # Define the system message template
    system_template = """The provided {context} is a wikipedia content make sure you 
    retrieve relevant information and stay true, 
    accurate and provide links of the webpages you got the most revelant information
    {context}"""

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False,combine_docs_chain_kwargs={"prompt": qa_prompt},memory=memory,verbose=True)

    return qa


def pure_llm_local(prompt):

    pipe = pipeline("text-generation", model=os.path.join(BASE_DIR,"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"), torch_dtype=torch.bfloat16, device_map="auto")

    messages = [
        {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": prompt},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    print(outputs[0]["generated_text"])
    # <|system|>
    # You are a friendly chatbot who always responds in the style of a pirate.</s>
    # <|user|>
    # How many helicopters can a human eat in one sitting?</s>
    # <|assistant|>
    return outputs[0]["generated_text"]

def pure_llm_api(prompt):

    fireworks.client.api_key = DevConfig.FIREWORKS_API_KEY
    completion = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/llama-v2-70b-chat",
    messages=[
        {
        "role": "user",
        "content": prompt,
        }
    ],
    # stream=True,
    n=1,
    max_tokens=512,
    temperature=0.1,
    top_p=0.9, 
    )
    return completion

def rag_api():

    fireworks.client.api_key = DevConfig.FIREWORKS_API_KEY
    llm = fireworks.client.ChatCompletion.create(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    messages=[
        {
        "role": "user",
        "content": '',
        }
    ],
    # stream=True,
    n=1,
    max_tokens=150,
    temperature=0.1,
    top_p=0.9, 
    )

    csv_docs = get_csv_splits(csv_file_path)

    embeddings = embed_fn()

    embed_index(doc_list=csv_docs,
                embed_fn=embeddings,
                index_store=DB_FAISS_PATH)
    


    docsearch = FAISS.load_local(DB_FAISS_PATH, embeddings)

    retriever=docsearch.as_retriever(search_kwargs={"k": 20})

    # Define the system message template
    system_template = """The provided {context} is a tabular dataset containing a english words and their meanings.
    The dataset includes the following columns:
    'id': unique id of the English word,
    'title': English word ,
    "meaning": the meaning of the English word in English, make sure you respond using English language to each query.
    ----------------
    {context}"""

    # system_template=[
    #     {
    #     "role": "user",
    #     "content": prompt,
    #     }
    # ],

    # Create the chat prompt templates
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, return_source_documents=False,combine_docs_chain_kwargs={"prompt": qa_prompt},memory=memory,verbose=True)

    return qa
   

  
def search_bot(prompt,scope): 
    response = None
    if scope == 'local':
        kb_engine=wikipedia(prompt=prompt)
        response = kb_engine.run(
        {
            "question": prompt
        }) 
    elif scope == 'api':
        kb_engine=rag_api()
        response = kb_engine.run(
        {
            "question": prompt
        }) 
    return response