from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
import gradio as gr
import os
os.environ["LANGCHAIN_RETURN_SOURCES"] = "true"
from googlesearch import search
import requests
from bs4 import BeautifulSoup

DB_FAISS_PATH="vectorstore/db_faiss"


def get_non_social_media_urls(query, num_results=3, sleep_interval=2):
    result = search(query, num_results=num_results, sleep_interval=sleep_interval)

    excluded_domains = ['linkedin.com', 'twitter.com', 'youtube.com', 'klu.ai', 'instergram.com', 'facebook.com']
    non_social_media_urls = [url for url in result if all(domain not in url for domain in excluded_domains)]
    return non_social_media_urls

def scrape_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
        
def extract_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract and print all text content
    text_content = soup.get_text()
    # print(text_content)
    return text_content
            
def retriever(query):
    num_results = 10
    sleep_interval = 2
    
    non_social_media_urls = get_non_social_media_urls(query, num_results, sleep_interval)
    
    for url in non_social_media_urls:
        page_content = scrape_url(url)
        
        if page_content:
           data= extract_text(page_content)
    return data

class Document:
        def __init__(self, page_content):
            self.page_content = page_content
            self.metadata={}

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=512, chunk_overlap=16, length_function=len):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function

    def split_documents(self, text):
        chunks = []
        start = 0
        end = self.chunk_size

        while start < self.length_function(text):
            chunk = text[start:end]
            chunks.append(Document(chunk))
            start = end - self.chunk_overlap
            end = start + self.chunk_size

        return chunks
    

def text_chunk(query):
    
    data=retriever(query)
    #split the text into chunks
    
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap  = 16, length_function=len)
    return text_splitter.split_documents(data)


def knowledg_base(DB_FAISS_PATH, query):

    
    text_chunks=text_chunk(query)
    
    embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-dot-v1')


    docsearch = FAISS.from_documents(text_chunks, embedding)
    docsearch.save_local(DB_FAISS_PATH)

    return docsearch

def qa_ollama(DB_FAISS_PATH, query):
   
    SYSTEM_PROMPT = """Write an accurate and concise answer (less than 100 words) for a given question using only the provided Search Result. You must only use information from the provided search results. Use an unbiased and journalistic tone. Today's date:Tuesday, 02 February, 2024. Use today's date to keep track of time. Combine search results together into a coherent answer. Do not repeat text. Cite search result using sources and references provided. Cite the most relevant results that answer the question. Show the sorce cited at the end. Do not cite irrelevant results. If different results refer to different entities with the same name, write separate answer for each entity. Do not make up the answers, source citions and references. """
     
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"  
    
    SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
    
    instruction = """
    {context}
    
    Question: {question}
    """   
    template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm=ChatOllama(model="llama2", temperature=0.9)

    docsearch=knowledg_base(DB_FAISS_PATH, query)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k":3,"top_k":3,"fetch_k":5, "distance_metric":"cos", "maximal_marginal_relevance":True}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    )
    
    return qa_chain


def call_ollama(user_input):
    qa_chain=qa_ollama(DB_FAISS_PATH,user_input)
    chat_history = []
    result=qa_chain.invoke({'query':user_input, "chat_history": chat_history})
    chat_history.append((user_input, result["result"]))
    return result['result']

# def main():
#     DB_FAISS_PATH="vectorstore/db_faiss"
       
#     def predict(message, history):
        
        
#         qa_chain=qa_ollama(DB_FAISS_PATH, message)
        
#         chat_history = []
        
        
#         result=qa_chain.invoke({'query':message, "chat_history": chat_history})
#         chat_history.append((message, result["result"]))
#         yield result['result']
    
#     gr.ChatInterface(predict).launch(share=True)

# if __name__=="__main__":
#     main()
    
    
    