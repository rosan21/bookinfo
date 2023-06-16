from apikey import apikey
import streamlit as st
import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = apikey

#App framework
st.title('🦜🔗 Youtube Script Creator')
prompt = st.text_input('Plug in your prompt here')

llm = OpenAI(temperature=0.9)

title_template = PromptTemplate(
    input_variables= ['topic'],
    template= 'Search for a appropriate heading for title {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikiresearch'],
    template = 'write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikiresearch}'
)
#memory
title_memory = ConversationBufferMemory(input_key= 'topic', memory_key='chat_history')
essay_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

title_chain = LLMChain(llm = llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm = llm , prompt= script_template, verbose=True, output_key='essay', memory=essay_memory)

wiki = WikipediaAPIWrapper()

if prompt:
   title = title_chain.run(prompt)
   wiki_research = wiki.run(prompt)
   script = script_chain.run(title = title, wikiresearch = wiki_research)
   
   st.write(title)
   st.write(script)
   
   with st.expander('Title History'):
        st.info(title_memory.buffer)

   with st.expander('Script History'):
        st.info(essay_memory.buffer)

   with st.expander('Wiki History'):
        st.info(wiki_research)
