# Importing all Dependencies
import os
import requests
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from streamlit_lottie import st_lottie
# from langchain.agents import create_pandas_dataframe_agent -- deprecated
# https://github.com/langchain-ai/langchain/discussions/11680
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# pip install -U langchain-community
# from langchain.vectorstores import FAISS --STB deprecated
from langchain_community.vectorstores import FAISS
# from langchain.llms import OpenAI --STB deprecated
from langchain_community.llms import OpenAI

# insert lottie file
url = "https://lottie.host/6705b87a-1078-4193-911c-87cef0f82c3c/YDpXLZjQMX.json"
response = requests.get(url)

# Display the Lottie animation
st_lottie(response.json(), width=150, height=150)


# Define function to handle PDF file upload and text extraction
def process_pdf(file):
    reader = PdfReader(file)
    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    return docsearch, chain

# Define Streamlit app function


def app():
    st.title('DataWhiz')
    st.write(
        ':computer: Effortlessly Extract Insights from PDF and CSV Files with DataWhiz.')
    with st.sidebar:
        st.header('Instructions')
        st.write('1. Enter your OpenAI API key.')
        st.write('2. Choose the file type betweeen a PDF or CSV file.')
        st.write('3. Upload your file(s).')
        st.write('4. Enter your questions and get answers.')
        st.write('5. Enjoy extracting insights!')

    key = st.text_input('Enter your OpenAI API key:')
    # OpenAI API Key
    os.environ['OPENAI_API_KEY'] = key

    option = st.selectbox("Select an option", ["PDF", "CSV"])
    file = st.file_uploader(f"Upload {option} file", type=[option.lower()])
    if file is not None:
        if option == "PDF":
            docsearch, chain = process_pdf(file)

            i = 0
            while True:
                i += 1
                query = st.text_input(
                    f'Enter your question {i}:', key=f'question_{i}')
                if not query:
                    break

                docs = docsearch.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
                st.write("Answer:", response)

        elif option == "CSV":
            df = pd.read_csv(file)
            agent = create_pandas_dataframe_agent(
                OpenAI(temperature=0), df, verbose=True)

            i = 0
            while True:
                i += 1
                query = st.text_input(
                    f'Enter your question {i}:', key=f'question_{i}')
                if not query:
                    break

                response = agent.run(query)
                st.write("Answer:", response)


if __name__ == '__main__':
    app()
st.write('DataWhiz can make mistakes. Consider checking important information.')
st.write('Made by Harshita Verma')
