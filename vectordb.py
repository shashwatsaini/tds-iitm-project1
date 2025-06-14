import bs4
import pickle
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

TDS_COURSE_PAGE_SCRAPED_FILE = 'tds_course_page_scraped.pkl'
TDS_DISCOURSE_PAGE_SCRAPED_FILE = 'tds_discourse_page_scraped.pkl'

def vectordb_init(llm, embeddings, index, vector_store):
    """
        Initializes Chroma DB. Reads the pickle files that contain scraped URLs and page contents, and adds them to the DB.
    """

    BATCH_SIZE = 10

    with open(TDS_COURSE_PAGE_SCRAPED_FILE, 'rb') as f:
        visited_pages = pickle.load(f)

    with open(TDS_DISCOURSE_PAGE_SCRAPED_FILE, 'rb') as f:
        visited_pages = visited_pages | pickle.load(f)

    urls = list(visited_pages.keys())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i:i+BATCH_SIZE]

        docs = [
            Document(page_content=visited_pages[url], metadata={"source": url})
            for url in batch_urls
        ]

        # Split and add to vector store
        all_splits = text_splitter.split_documents(docs)

        if all_splits:
            _ = vector_store.add_documents(all_splits)
            print(f"Added batch {i // BATCH_SIZE + 1} with {len(all_splits)} chunks")
        else:
            print(f"Batch {i // BATCH_SIZE + 1} was empty or failed")

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    return