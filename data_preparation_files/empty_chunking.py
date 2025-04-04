#Imports
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import time
import requests
import gdown
import os
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import pymupdf4llm
import os
from io import BytesIO
import base64
from openai import OpenAI
from PyPDF2 import PdfReader
import re
from PIL import Image
load_dotenv("credentials.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Chunk the files and add onto Chroma DB
df = pd.read_csv('empty_conv2md.csv')
failed_df = pd.DataFrame()
long_context_list = []
paragraph_list = []

for index, row in df.iterrows():
    try:
        # get the md file path
        if 'data/' in row['md_path']: 
            md_file_path = row['md_path']
        else:
            md_file_path = 'data/' + row['md_path']
        # read the md file
        with open(md_file_path, 'r',encoding="utf-8") as file:
            # read the file
            data = file.read()
    
        # create file as a document
        # metadata columns: year	location	format	title	isic_section	isic_division_author	type	theme	topic	file_link	source_domain	pdf_path	md_path
        longcontext_doc = Document(page_content=data,metadata={
            'year': row['year'],
            'location': row['location'],
            'format': row['format'],
            'title': row['title'],
            'isic_section': row['isic_section'],
            'isic_division_author': row['isic_division_author'],
            'type': row['type'],
            'theme': row['theme'],
            'topic': row['topic'],
            'file_link': row['file_link'],
            'pdf_path': row['pdf_path'],
            'md_path': md_file_path
        })
        long_context_list.append(longcontext_doc)

        # split the document into list of paragraphs
        # split by ----- first
        split_data = data.split('-----')
        for page in split_data:
            # split page further
            text_splitter = CharacterTextSplitter(chunk_size = 1800, chunk_overlap=200, separator='', strip_whitespace=False)
            chunks = text_splitter.split_text(page)
            for chunk in chunks:
                para_doc = Document(page_content=chunk,metadata={
                    'year': row['year'],
                    'location': row['location'],
                    'format': row['format'],
                    'title': row['title'],
                    'isic_section': row['isic_section'],
                    'isic_division_author': row['isic_division_author'],
                    'type': row['type'],
                    'theme': row['theme'],
                    'topic': row['topic'],
                    'file_link': row['file_link'],
                    'pdf_path': row['pdf_path'],
                    'md_path': md_file_path
                })
                paragraph_list.append(para_doc)
    except Exception as e:
        fail = pd.DataFrame(row)
        failed_df = pd.concat([failed_df,fail],ignore_index=True)
        print(e)
        print(row['file_link'])
        continue

try:
    # embed the paragraph_list into Chroma db
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    chunk_vector_store = Chroma(
        collection_name="chunk_vector_store",
        embedding_function=embeddings,
        persist_directory="./chunk_vector_store",  # Path for Chromadb
    )

    longcontext_vector_store = Chroma(
        collection_name="longcontext_vector_store",
        embedding_function=embeddings,
        persist_directory="./longcontext_vector_store", # Path for Chromadb
    )
except Exception as e:
    print(e)
    print('Error in initialising Chroma db')

fail = pd.DataFrame()
try:
    # add the chunk_doc into Chroma db
    for para in paragraph_list:
        try:
            chunk_vector_store.add_documents([para])
            chunk_vector_store.persist()
        except Exception as e:
            row = pd.DataFrame({'file_link':para.metadata['file_link'],'method':'chunk'})
            fail = pd.concat([fail,row],ignore_index=True)
            print(e)
            print('Error in adding documents into Chroma db')
            print(para.metadata['file_link'])
    print('Done embedding paragraphs')

    # add the longcontext_doc into Chroma db
    for doc in long_context_list:
        try:
            longcontext_vector_store.add_documents([doc])
            longcontext_vector_store.persist()
        except Exception as e:
            row = pd.DataFrame({'file_link':doc.metadata['file_link'],'method':'longcontext'})
            fail = pd.concat([fail,row],ignore_index=True)
            print(e)
            print('Error in adding documents into Chroma db')
            print(doc.metadata['file_link'])
    print('Done embedding longcontext')
except Exception as e:
    print(e)
    print('Error in adding documents into Chroma db')