import os 
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import dotenv_values
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import streamlit as st

st.title("A fun project with Langchain for Q&A from a PDF")


config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"]=config["OPENAI_KEY"]

# Read the PDF File 
pdfreader = PdfReader('pro-mern-stack-full-stack.pdf')


# Read the content of the file and store as a string 
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content


# We need to split the text using Character Text Split such that it sshould not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()


document_search = FAISS.from_texts(texts, embeddings)


chain = load_qa_chain(OpenAI(), chain_type="stuff")


input_Text = st.text_input("Enter the question you want to search : ")

if input_Text:
    docs = document_search.similarity_search(input_Text)
    value  = chain.run(input_documents=docs, question=input_Text)
    st.write(value)