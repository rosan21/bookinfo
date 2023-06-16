import os
from apikey import apikey

os.environ['OPENAI_API_KEY'] = apikey



# from langchain.document_loaders import DirectoryLoader

# directory = 'C:/Users/user/Desktop/AutoGPT/data'

# def load_doc(directory):
#     loader = DirectoryLoader(directory)
#     documents = loader.load()
#     return documents

# documents = load_doc(directory)

# from langchain.text_splitter import RecursiveCharacterTextSplitter

# def split_docs(documents, chunk_size=1000, chunk_overlap=20):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     docs = text_splitter.split_documents(documents)
#     return docs

# docs = split_docs(documents)

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
 
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key='b988fddd-4694-4ed7-ae36-2bc1afadf0b1',
    environment='us-west4-gcp'
)

index_name = 'langchain-demo'

docsearch = Pinecone.from_existing_index(index_name, embeddings)

def get_similar_docs(query, k=2, score = False):
    if score:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    else:
        similar_docs= docsearch.similarity_search(query, k=k)
    return similar_docs


from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents = similar_docs, question = query)
    return answer

import streamlit as st
st.title('🦜🔗 Book informations')
query = st.text_input('Plug in your prompt here')

if query:
   answer = get_answer(query=query)
   st.write(answer)