import os
import hashlib
import pickle
from typing import Set
from pydantic import Field
from crewai import Agent, Task
from crewai_tools import RagTool, ScrapeWebsiteTool
from scrape import scrape_tds_course_page, scrape_tds_discource_page, query_tds_course_page, query_tds_discourse_page

HASH_CACHE_FILE = 'rag_content_hashes.pkl'

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_hash_cache():
    if os.path.exists(HASH_CACHE_FILE):
        with open(HASH_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return set()

def save_hash_cache(hash_cache):
    with open(HASH_CACHE_FILE, 'wb') as f:
        pickle.dump(hash_cache, f)

def return_rag_agent(llm):
    """
    Initialize and return the RAG agent configured to answer questions about the TDS course page.
    Avoids re-adding content already embedded using hash caches.
    """

    rag_config = {
        'embedder': {
            'provider': 'google',
            'config': {
                'model': 'models/text-embedding-004',
                'task_type': 'RETRIEVAL_DOCUMENT'
            }
        }
    }

    # Custom implementation of RagTool to save retrieved documents
    class MyRagTool(RagTool):
        retrieved_docs: Set[str] = Field(default_factory=set)

    rag_tool = MyRagTool(config=rag_config, llm=llm)

    original_run = rag_tool._run

    def debug_run(*args, **kwargs):
        result = original_run(*args, **kwargs)

        # Save the URLs of the docs that are retrieved
        url = query_tds_course_page(result.removeprefix('Relevant Content:\n'))

        if url is not None:
            rag_tool.retrieved_docs.add(url)
        else:
            url = query_tds_discourse_page(result.removeprefix('Relevant Content:\n'))
            if url is not None:
                rag_tool.retrieved_docs.add(url)

        # print("\n[DEBUG] Retrieved Document Chunks:\n", result)
        return result

    rag_tool._run = debug_run

    # Load existing hashes
    existing_hashes = load_hash_cache()
    new_hashes = set()

    # Scrape the course page
    scraped_pages = scrape_tds_course_page()
    for url, content in scraped_pages.items():
        h = content_hash(content)
        if h not in existing_hashes:
            print(f"Adding new page to RAG: {url}")
            
            combined = f"[URL] {url}\n\n{content}"
            rag_tool.add('text', combined)

            new_hashes.add(h)
        else:
            print(f"Skipped (already added): {url}")


    # Scrape the discourse page
    scraped_pages = scrape_tds_discource_page()
    for url, content in scraped_pages.items():
        h = content_hash(content)
        if h not in existing_hashes:
            print(f"Adding new page to RAG: {url}")
            
            combined = f"[URL] {url}\n\n{content}"
            rag_tool.add('text', combined)

            new_hashes.add(h)
        else:
            print(f"Skipped (already added): {url}")

    # Save updated hashes
    save_hash_cache(existing_hashes.union(new_hashes))

    scrape_tool = ScrapeWebsiteTool()

    # Setup RAG agent and task
    rag_agent = Agent(
        role="RAG Expert",
        goal="Answer questions using RAG from the course webpage. You must use the provided tool at least thrice. You must use the scrape tool is an additional link is provided.",
        backstory="You are an expert at understanding and extracting insights from course pages. Use the provided tools to do this. You must use the scrape tool is an additional link is provided.",
        llm=llm,
        tools=[rag_tool, scrape_tool],
        description='An agent that can answer questions about the TDS course page using RAG.',
        verbose=False
    )

    rag_task = Task(
        name='rag_task',
        agent=rag_agent,
        description='Answer this question- {question} about the Tools for Data Science (TDS) course page using RAG.',
        expected_output='A detailed answer to the question- {question} based on the Tools for Data Science (TDS) course page content.',
    )

    return rag_tool, rag_agent, rag_task

def return_context_agent(llm):
    scrape_tool = ScrapeWebsiteTool()

    context_agent = Agent(
        role='Context Agent',
        goal='Summarize the content of the given webpage ({url}) in context to the question asked: "{question}". Use the provided tool to retrieve and analyze the webpage. Ensure the summary is a single readable line with no extra line breaks or excessive whitespace.',
        backstory='You are an expert at analyzing and summarizing webpages. Your job is to extract and summarize only the content relevant to the question: "{question}", based on the webpage at {url}.',
        llm=llm,
        tools=[scrape_tool],
        description='An agent that summarizes relevant information from a webpage in a single, readable line without line breaks or unnecessary whitespace.',
        verbose=False
    )

    context_task = Task(
        name='context_task',
        agent=context_agent,
        description=(
            'Scrape and summarize relevant content in a SINGLE LINE from the URL: {url} in response to the question: "{question}". '
            'Avoid using \\n characters or excessive whitespace. Keep it clean and readable.'
        ),
        expected_output='A one-line, human-readable summary of the content on the URL that answers or adds context to the question: "{question}", without any extra spacing or line breaks.'
    )

    return context_agent, context_task

