from langchain.llms import OpenAI
from dotenv import dotenv_values
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st


# Load the .env Files
config = dotenv_values(".env")

# Setting up the openAPI key in OS Environment
os.environ["OPENAI_API_KEY"] = config["OPENAI_KEY"]

# Streamlit Framework
st.title("A fun project with Langchain")
input_Text = st.text_input("Enter the text you want to search : ")

# Initialize the LLM Model
llm = OpenAI(temperature=0.8)


# Memory
person_memory = ConversationBufferMemory(input_key="name", memory_key="chat_history")
movie_name_memory = ConversationBufferMemory(
    input_key="person", memory_key="chat_history"
)
actor_memory = ConversationBufferMemory(
    input_key="movie_name", memory_key="chat_history"
)


# First Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=["name"], template="Tell me about {name} in about 1000 words"
)

# Initialize the First LLM Chain
chain = LLMChain(
    llm=llm,
    prompt=first_input_prompt,
    verbose=True,
    output_key="person",
    memory=person_memory,
)


# Second Prompt Template
second_input_prompt = PromptTemplate(
    input_variables=["person"], template="what all the movies/ads {person} has acted?"
)

# Initialize the Second LLM Chain
chain2 = LLMChain(
    llm=llm,
    prompt=second_input_prompt,
    verbose=True,
    output_key="movie_name",
    memory=movie_name_memory,
)


# Third Prompt Templates
third_input_prompt = PromptTemplate(
    input_variables=["movie_name"],
    template="list down the leading co-actors acted with {movie_name}",
)

# Third Chain
chain3 = LLMChain(
    llm=llm,
    prompt=third_input_prompt,
    verbose=True,
    output_key="actors",
    memory=actor_memory,
)


# Initialize the simple Sequntial Chain for calling multiple chains
# parent_chain = SimpleSequentialChain(chains=[chain,chain2,chain3],verbose=True)

# this Method we have to use in case of SimpleSequentialChain
# if input_Text:
#     st.write(parent_chain.run(input_Text))

# Initialize the sequential chain
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=["name"],
    output_variables=["person", "movie_name", "actors"],
    verbose=True,
)

# this Method we have to use in case of SequentialChain

if input_Text:
    st.write(parent_chain({"name": input_Text}))

    with st.expander("Person Name"):
        st.info(person_memory.buffer)
    with st.expander("Movies Name"):
        st.info(movie_name_memory.buffer)
    with st.expander("Co-Actors Name"):
        st.info(actor_memory.buffer)
