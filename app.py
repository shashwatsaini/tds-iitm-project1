import os
import base64
from flask import Flask, request, jsonify
from crewai import Crew, Process, LLM
from agents import return_rag_agent, return_context_agent

app = Flask(__name__)

# Base LLM config
llm = LLM(
    model='gemini/gemini-2.0-flash',
    api_key=os.environ['GEMINI_API_KEY']
)

# Load agents
rag_tool, rag_agent, rag_task = return_rag_agent(llm)
context_agent, context_task = return_context_agent(llm)


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

        # Step 1: Run RAG crew
        crew = Crew(
            name='crew',
            agents=[rag_agent],
            tasks=[rag_task],
            process=Process.sequential,
            description='RAG crew to answer TDS questions.',
            verbose=False
        )


        if link:
            rag_result = crew.kickoff(inputs={"question": question + f' helpful link: {link}'})
        else:
            rag_result = crew.kickoff(inputs={"question": question})

        # Step 2: Context crew for each document retrieved
        context_crew = Crew(
            name='context_crew',
            agents=[context_agent],
            tasks=[context_task],
            process=Process.sequential,
            description='Crew to summarize URLs in ONE LINE.',
            verbose=False
        )

        link_summaries = []
        for url in rag_tool.retrieved_docs:
            output = context_crew.kickoff(inputs={"question": question, "url": url})

            if output.tasks_output and len(output.tasks_output) > 0:
                summary = output.tasks_output[0].raw.strip().replace('\n', ' ')
            else:
                summary = "No summary available."

            link_summaries.append({
                "url": url,
                "text": summary
            })

        return jsonify({
            "answer": rag_result.raw,
            "links": link_summaries
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
