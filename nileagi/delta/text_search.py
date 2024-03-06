from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.retrievers import WikipediaRetriever, ArxivRetriever
from langchain_core.prompts import PromptTemplate

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import sys

from langchain_community.chat_models import ChatOllama
import gradio as gr

DB_FAISS_PATH="vectorstore/db_faiss"

import os
os.environ["LANGCHAIN_RETURN_SOURCES"] = "true"


def search_wiki(question):
    wiki_doc=WikipediaRetriever()
    return wiki_doc.get_relevant_documents(query=question) 

def search_arxiv(question):
    arxiv_doc=ArxivRetriever()
    return arxiv_doc.get_relevant_documents(query=question)
 
def search_duckduckgo(question):
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time="d", max_results=3)
    search=DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
    ddg_doc=search.run(question)
    return ddg_doc

def docs_retriever(question):
    ddg_doc=search_duckduckgo(question)
    
    wiki_doc=search_wiki(question)
    wiki_doc_str=str(wiki_doc)
    arxiv_doc=search_arxiv(question)
    arxiv_doc_str=str(arxiv_doc)
    
    docs_retriever=ddg_doc+wiki_doc_str+arxiv_doc_str
    return docs_retriever


def text_chunk(question):
    data=docs_retriever(question)
    #split the text intio chunks
    
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap  = 16, length_function=len, is_separator_regex=False,)
    return data,text_splitter.split_text(data)

def knowledg_base(DB_FAISS_PATH, question):

    data,text_chunks=text_chunk(question)
    #Dowload Sentence Transformers Embedding from Huggging Face\
    
    embedding= HuggingFaceEmbeddings(model_name='sentence-transformers/multi-qa-MiniLM-L6-dot-v1')

    #Converting the text Chunks into embeddings and saving the embeddings into FAISS knowledge Base

    docsearch = FAISS.from_texts(text_chunks, embedding)
    docsearch.save_local(DB_FAISS_PATH)

    return data,docsearch

def qa_ollama(DB_FAISS_PATH, question):
    
    # SYSTEM_PROMPT = """You are a Search Engine, answer the question precisely, correctly , shortly and include references and page source to your answer.Don't try to make up the answer if you don't know the answer."""
    
    # SYSTEM_PROMPT = """You are a Search Engine,Read all these documents and use only relevant and recently information from those documents to answer the users question in a very concise way. In your answer write it like academic or journalist would write it by make sure you have a supportive citation and supportive link"""
    
     
    # SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
    #     Use the following pieces of retrieved context from wikipedia to answer the question. \
    #     If you don't know the answer, just say that you don't know. \
    #     Write an accurate and concise answer in less than 100 words for a given question. \
    #     Use an unbiased and journalistic tone. \
    #     Today date is 8th February, 2024 . \
    #     Combine search results together into a coherent answer. \
    #     Do not repeat text. \
    #     For any sentence that have suportive link, provide the link as Cition in your answer in form of [link number]. \
    #     Show the url source of your citations with respect to citation number you provide, at the end of your answer. \
    #     If different results refer to different entities with the same name, write separate answer for each entity. \
    #     Do not make up the answers, provide url of page source at the end of your answer. """
        
        
    SYSTEM_PROMPT = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, \
        answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

        Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
        justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
        that justify the answer. Use the following format for your final output:

        <cited_answer>
            <answer></answer>
            <citations>
                <citation><source_id></source_id><url></url></citation>
                <citation><source_id></source_id><url></url></citation>
                ...
            </citations>
        </cited_answer>

        Here are the Wikipedia articles:{context}"""
     
    
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

    data,docsearch=knowledg_base(DB_FAISS_PATH, question)

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k":3,"top_k":3,"fetch_k":5,"top_p":1, "distance_metric":"cos", "maximal_marginal_relevance":True}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
    )
    
    # retriever = db.as_retriever(): This line initializes a retriever using a database object db.
    # retriever.search_kwargs["distance_metric"] = "cos": It sets the distance metric for the retriever's search to cosine similarity.
    # retriever.search_kwargs["fetch_k"] = 20: It specifies the number of documents to fetch during the search.
    # retriever.search_kwargs["maximal_marginal_relevance"] = True: This enables the maximal marginal relevance for the retriever's search results.
    # retriever.search_kwargs["k"] = 20: It sets the value of k for the retriever, which is used to specify the number of documents to fetch.
    
    # "fetch_k": This parameter determines the number of documents to be fetched from the vector database during the search process. For example, setting "fetch_k" to 20 would retrieve the top 20 most relevant documents based on the search query and other parameters
    # "k": In the context of LangChain-based apps, "k" represents the most similar embeddings to be retrieved. For instance, setting "k" to 10 would mean that the search should return the top 10 most similar embeddings
    
    return data,qa_chain

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

def ollama_call_text_to_text(user_input):
    _, qa_chain=qa_ollama(DB_FAISS_PATH,user_input)
    chat_history = []
    result=qa_chain.invoke({'query':user_input, "chat_history": chat_history})
    chat_history.append((user_input, result["result"]))
    return result['result']

# def main(): 
#     DB_FAISS_PATH="vectorstore/db_faiss"
#     chat_history=[]
#     while True:
#         user_input=input(f"ask: ")
#         if user_input=='exit':
#             print('Exiting')
#             sys.exit()
#         if user_input=='':
#             continue
#         _,qa_chain=qa_ollama(DB_FAISS_PATH, user_input)
#         result=qa_chain.invoke({'query':user_input, 'chat_history':chat_history})
#         chat_history.append((user_input, result['result']))
#         # source_url=[data[i].metadata['source'] for i in range(len(data))]
#         # print(f"Answer:{result['result']}")
#         print(result['result'])
#         print('\n')
#         # print('Page Source','\n', source_url[0])
#         # # print('\n')
#         # print('See Also', '\n', source_url[1], '\n', source_url[2])
#         # print('\n\n\n')
        
# if __name__=='__main__':
#     main()