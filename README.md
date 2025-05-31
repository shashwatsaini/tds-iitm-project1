# RAG-Powered Q\&A System with Crew\.ai and Chroma DB
### *For IITM Tools in Data Science - Project 1*

## Overview

This project implements a Retrieval-Augmented Generation (RAG) based question-answering system using Crew\.ai agents, combined with a Chroma vector database for document retrieval and contextual summarization.

### Key Features and Approach

1. **Data Collection & Caching**
   * Data is scraped from course and discourse pages and cached locally using a pickle file to avoid repeated scraping and speed up processing.
   * Discourse is scrapped via logging in with a student account, as the discourse is limited to verified student IDs.

2. **RAG Setup with Crew\.ai and Chroma DB**

   * The cached data is loaded into Crew\.ai's RAG tool, which integrates with a Chroma vector database.
   * The Chroma DB is updated only when a new entry is detected using hash caches, ensuring efficient incremental updates.

3. **Question Handling**

   * User input (question + optional link) is fed to the RAG crew.
   * The RAG crew uses the vector DB to retrieve relevant documents and formulate an initial answer.
   * The retrieved documents’ URLs are extracted for further processing.

4. **Contextual Enrichment**

   * Each retrieved URL is processed by a separate "context crew" agent.
   * This agent provides a concise one-line summary for each document, adding more context to the answer.

5. **JSON Response Output**  
   The final API response contains:

   * The generated answer to the user’s question.
   * A list of URLs with their corresponding one-line summaries, giving transparent context to the solution.

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

