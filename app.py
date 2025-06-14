import json

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from flask import Flask, request, jsonify
from flask_cors import CORS

from vectordb import vectordb_init
from graphs import graph_1, graph_2

app = Flask(__name__)
cors = CORS(app)

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

vectordb_init(llm, embeddings, index, vector_store)
basic_answer_graph = graph_1(llm, vector_store)
refinement_graph = graph_2(llm)

@app.route('/api/', methods=['POST'])
def api_handler():
    try:
        data = request.get_json()

        # Required field
        question = data.get('question', '').strip()
        if not question:
            return jsonify({"error": "Missing 'question' field."}), 400
        
        # Optional field
        link = data.get('link', '').strip()

        # Step 1: Run Graph 1 - retrieve and generate initial answer
        initial_state = {"question": question}
        result = basic_answer_graph.graph.invoke(initial_state)
        initial_answer = result["answer"]

        # Step 2: Refine using documents in the latest retrieval log
        link_objects = []

        for doc in basic_answer_graph.retrieval_log[-1]["retrieved"][:2]:
            refine_result = refinement_graph.graph.invoke({
                "question": question,
                "answer": initial_answer,
                "doc": doc
            })

            try:
                # Parse string response to JSON
                refined_text = refine_result['refined_answer']
                url = doc.metadata['source']

                if refined_text and url:
                    link_objects.append({
                        "url": url,
                        "text": refined_text
                    })
                    
            except Exception as e:
                print(e)

        return jsonify({
            "answer": initial_answer,
            "links": link_objects
        })        

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return '✅ App is up and running!', 200

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
