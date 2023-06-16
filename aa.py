import os
from apikey import apikey

os.environ['OPENAI_API_KEY'] = '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'

urls = [
    'https://bau.edu/blog/application-process-for-us-universities/',
    'https://www.studyusa.com/en/a/34/applying-for-admission-to-a-u-s-program',
    'https://www.crimsoneducation.org/nz/blog/how-to-apply-to-us-universities/',
    'https://leverageedu.com/blog/admission-process-to-study-in-usa/',
    'https://openmynetwork.com/LEEP.php',
    'https://openmynetwork.com/Test271828/m/faq/browse/recent',
    'https://openmynetwork.com/Test271828/m/faq/browse/recent?page=2&per_page=14'

]

from langchain.document_loaders import UnstructuredURLLoader
loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n', 
                                      chunk_size=1000, 
                                      chunk_overlap=200)


docs = text_splitter.split_documents(data)


#Embeeding in Vectorstorage
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

##Embedding the doc in vectore storage and saving the file

# vectorStore_openAI = FAISS.from_documents(docs, embeddings)

# with open("faiss_store_openai.pkl", "wb") as f:
#     pickle.dump(vectorStore_openAI, f)

#Reading from the file we created i.e. vectore storage
with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

#Retriving the data from the docs and feeding it to the llm
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm=OpenAI(temperature=0.9)

chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

chain({"question": "How to apply for scholarship?"}, return_only_outputs=True)

import streamlit as st

#App Framework
st.title("🦜🔗 Document Details")
prompt = st.text_input('Plug in the prompt here')

if prompt:
    answer = chain({"question": prompt}, return_only_outputs=True)
    st.write(answer['answer'])

    