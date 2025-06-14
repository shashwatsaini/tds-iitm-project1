# RAG-Powered Q\&A System with Langchain and Chroma DB
### *For IITM Tools in Data Science - Project 1*

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based question-answering system using **LangChain**, integrated with a **Chroma vector database** for document retrieval and contextual reasoning. It is tailored for IITM’s Tools in Data Science course, enabling users to query course-related content with high relevance and transparency.

---

## Key Features and Approach

1. **Data Collection & Caching**
   - Course-related content from official course and discourse pages is scraped and cached locally using pickle files.
   - Discourse scraping is authenticated using student credentials, as content access is restricted to verified users.
   - The cached content avoids redundant scraping and accelerates system performance.

2. **RAG Pipeline with LangChain and Chroma DB**
   - Cached documents are converted into embeddings using a LangChain-compatible embedding model.
   - The embeddings are stored in a **Chroma vector store**, with smart hash-based checks to ensure incremental updates without duplication.
   - The LangChain RAG pipeline uses this vector store to retrieve relevant chunks for a given question.

3. **Question Handling via LangChain Chain**
   - The incoming user question (optionally with a reference link) is passed into a LangChain RAG chain.
   - The chain retrieves the most relevant documents using similarity search and generates an answer using an LLM via LangChain’s integration.
   - Retrieved source documents are tracked to extract their originating URLs.

4. **Contextual Enrichment with URL Mapping**
   - Each retrieved document is mapped to its source URL and passed through a LangChain summarization chain.
   - This step yields a one-line context summary for each document, enhancing transparency and usefulness.

5. **Structured JSON API Response**
   - The output of the LangChain pipeline is returned as a JSON response that includes:
     - The generated answer from the RAG model.
     - A list of source URLs along with one-line summaries for contextual reference.

6. **Pickle-Based Storage of Scraped Pages and URLs**  
   - The retrieved URLs and page contents from scraping are stored locally in pickle files.  
   - To regenerate or refresh this data, simply run `scrape.py` again.  
   - This ensures fast data loading and avoids unnecessary scraping on subsequent runs.

---

## API Usage

**Endpoint:** `/api/`  
**Method:** POST  
**Request JSON:**

```json
{
  "question": "Your question here",
  "link": "Optional link here"
}
```

**Response JSON:**

```json
{
  "answer": "Generated answer text",
  "links": [
    {
      "url": "URL 1",
      "text": "One-line summary for URL 1"
    },
    ...
  ]
}
```

---

## Environment Variables & Credentials

To run this project, you need to provide the following sensitive information securely:

* **Google Gemini API Key:**
  Add your Google Gemini API key as an environment variable named `GEMINI_API_KEY`.
  This key is used by the LLM agent to process questions and generate answers.

* **Discourse Credentials:**
  The scraping of discourse pages requires login credentials.
  Add your Discourse email ID and password as environment variables named `DISCOURSE_EMAIL` and `DISCOURSE_PASSWORD` respectively.
  This is necessary because access to discourse content is restricted to verified student accounts.

* **Langsmith API Key:**
  Add your Langsmith API key as an environment variable named `LANGSMITH_API_KEY`.
  Langsmith can be used to track LLM calls more precisely.

