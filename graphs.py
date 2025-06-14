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

prompt = hub.pull("rlm/rag-prompt")

retrieval_log = []

class graph_1:
    """
        Graph 1: Retrieval-Augmented Generation (RAG) Graph

        This graph handles the initial retrieval and answer generation process.
        - Takes a user question as input.
        - Retrieves relevant documents from the vector store using similarity search.
        - Logs the retrievals for downstream use.
        - Passes the documents and question to an LLM to generate an initial answer.

        Nodes:
        - retrieve: Performs similarity search and logs retrieved documents.
        - generate: Uses LangChain prompt + LLM to create an answer from the documents.
    """
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        self.retrieval_log = []

        self.graph = self._build_graph()

    def _retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        self.retrieval_log.append({
            "question": state["question"],
            "retrieved": retrieved_docs
        })
        return {"context": retrieved_docs}

    def _generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def _build_graph(self):
        builder = StateGraph(self.State)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        return builder.compile()

class graph_2:
    """
        Graph 2: Answer Refinement Graph

        This graph refines an initial answer by grounding it in individual retrieved documents.
        - Takes the initial answer, question, and one document at a time.
        - Produces a refined explanation based only on that document.
        - This ensures the final context provided to the user is traceable and document-grounded.

        Nodes:
        - refine_one_doc: Refines the answer using a specific document’s content.
    """

    class RefineState(TypedDict):
        question: str
        answer: str
        doc: Document
        refined_answer: str

    def __init__(self, llm):
        self.llm = llm
        self.graph = self._build_graph()

    def _refine_one_doc(self, state: RefineState):
        doc_content = state["doc"].page_content
        refining_prompt = f"""
        Given the original answer:
        \"\"\"{state['answer']}\"\"\"

        And the following document:
        \"\"\"{doc_content}\"\"\"

        Please support the answer by only explaining from content in the provided document.
        Your output must be readable text, do not output brackets, escape sequences, html, or markdown.
        """

        response = self.llm.invoke(refining_prompt)
        return {"refined_answer": response.content}

    def _build_graph(self):
        builder = StateGraph(self.RefineState)
        builder.add_node("refine_one_doc", self._refine_one_doc)
        builder.set_entry_point("refine_one_doc")
        return builder.compile()
    