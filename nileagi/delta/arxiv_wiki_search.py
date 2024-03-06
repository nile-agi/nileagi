from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.retrievers import WikipediaRetriever,ArxivRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.prompts import PromptTemplate
import sys
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import CTransformers, ollama
from langchain_openai import OpenAIEmbeddings

DB_FAISS_PATH="vectorstore/db_faiss"

def search_wiki(query):
    # loader = WikipediaLoader(query=query, load_max_docs=2)
    # data=loader.load()
    retriever = WikipediaRetriever(load_max_docs=2)
    data=retriever.get_relevant_documents(query=query)
    
    return data, retriever

def search_arxiv(query):
    retriever = ArxivRetriever(load_max_docs=2)
    data = retriever.get_relevant_documents(query="1605.08386")
    return data, retriever

def search_arxiv_wiki(query):
    _, wiki_retriever=search_wiki(query)
    _, arxiv_retriever=search_arxiv(query)
    
    merger_retriever=MergerRetriever(retrievers=[wiki_retriever, arxiv_retriever])
    combine_data=merger_retriever.get_relevant_documents(query=query)
    
    return combine_data, merger_retriever

def text_chunk(query):
    data, _=search_arxiv_wiki(query)
    #split the text intio chunks
    
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(data)

def knowledg_base(DB_FAISS_PATH, query):

    text_chunks=text_chunk(query)
    #Dowload Sentence Transformers Embedding from Huggging Face
    # embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-dot-v1')

    #Converting the text Chunks into embeddings and saving the embeddings into FAISS knowledge Base

    docsearch = FAISS.from_documents(text_chunks, embedding)
    docsearch.save_local(DB_FAISS_PATH)

    return docsearch



def qa_llama(DB_FAISS_PATH, query):
    # SYSTEM_PROMPT = """Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    
    SYSTEM_PROMPT = """You are a Search Engine, answer the question precisely, correctly and include references to answer.Don't try to make up the answer if you don't know the answer."""
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"  
    
    SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
    
    instruction = """
    {context}
    
    Question: {question}
    """
    
    template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm=CTransformers(model="models/llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", config={"max_new_tokens":512, "temperature":0.1})
    
    docsearch=knowledg_base(DB_FAISS_PATH, query)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_type="mmr", search_kwargs={"k": 2, "include_metadata":True}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    
    )
    

    return qa_chain


def qa_ollama(DB_FAISS_PATH, query):
    # SYSTEM_PROMPT = """Use the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    
    SYSTEM_PROMPT = """You are a Search Engine, answer the question precisely, correctly , shortly and include references to answer.Don't try to make up the answer if you don't know the answer."""
    
    
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<>\n", "\n<>\n\n"  
    
    SYSTEM_PROMPT = B_SYS + SYSTEM_PROMPT + E_SYS
    
    instruction = """
    {context}
    
    Question: {question}
    """
    
    template = B_INST + SYSTEM_PROMPT + instruction + E_INST

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # llm=CTransformers(model="models/llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", config={"max_new_tokens":512, "temperature":0.1})
    llm=ChatOllama(model="mistral")

    docsearch=knowledg_base(DB_FAISS_PATH, query)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    )
    

    return qa_chain



def ollama_call_text_to_text(user_input):
    qa_chain=qa_ollama(DB_FAISS_PATH,user_input)
    chat_history = []
    result=qa_chain.invoke({'query':user_input, "chat_history": chat_history})
    chat_history.append((user_input, result["result"]))
    return result['result']

# def main():
#     while True:
#         user_input=input(f"-> **Ask:**  ")
#         if user_input=='exit':
#             print('Exiting')
#             sys.exit()
#         if user_input=='':
#             continue
#         qa_chain=qa_ollama(DB_FAISS_PATH, user_input)
        
#         chat_history = []
        
#         result=qa_chain.invoke({'query':user_input, "chat_history": chat_history})
        
#         chat_history.append((user_input, result["result"]))
        
#         print(f"**Answer**:  {result['result']}\n")
        
    
    
# if __name__=="__main__":
#     main()