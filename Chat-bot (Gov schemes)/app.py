import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# Import the vector database loader from our knowledge_base module
from src.knowledge_base import GovernmentSchemesDB

# Load local environment variables from the .env file
load_dotenv()

# Configure the Google API key from environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

# Initialize the Flask Web Server
app = Flask(__name__)

# Initialize the Government Schemes vector database 
# (This instantly loads the index from disk cache under 7 seconds)
schemes_db = GovernmentSchemesDB()


def query_gemini(prompt: str, api_key: str, model: str = "gemini-2.5-flash-lite") -> str:
    """
    Sends a synchronous HTTP POST request directly to the Google Generative Language API.
    By using standard library urllib, we keep dependencies lightweight and simple.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    # Configure request payload parameters
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2  # Low temperature makes output factual and reduces hallucinations
        }
    }
    
    # Encode payload to UTF-8 bytes
    req_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        # Perform synchronous network request
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            # Parse text response from the nested JSON candidate parts structure
            return res_json["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_json = json.loads(error_body)
            error_msg = error_json.get("error", {}).get("message", str(e))
        except Exception:
            error_msg = error_body
        raise Exception(f"Gemini API Error: {error_msg}")
    except Exception as e:
        raise Exception(f"Gemini Request failed: {str(e)}")


@app.route("/")
def index():
    """Renders the HTML landing page containing the chatbot user interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint called by frontend JavaScript.
    Queries the FAISS database, builds context, constructs a strict prompt to prevent
    hallucinations, invokes Gemini model, and returns the answer.
    """
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # 1. Retrieve the top 4 most semantically similar scheme documents using FAISS search
    retrieved_docs = schemes_db.search(user_message, k=4)

    # 2. Extract contents and build a clean textual context block
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
    context = "\n\n".join(context_parts) if context_parts else "No relevant documents found in the database."

    # 3. Build a system prompt that enforces factual grounding but permits reasoning for suggestions/improvements
    prompt = f"""You are a helpful and direct assistant that explains Indian government schemes.

Use the following context to answer the user's query:
{context}

User Query: {user_message}

Strict Retrieval-Augmented Generation (RAG) Instructions:
- For factual queries about the scheme (such as eligibility, benefits, required documents, or application process), rely ONLY on the provided context. If the context does not contain these facts, reply with: "I cannot find relevant information in the government schemes database." Do not invent scheme details.
- For queries asking for suggestions, criticisms, recommendations, or ways to improve a scheme, analyze the scheme's details from the context and generate logical, constructive, and helpful suggestions or reforms using your reasoning capability.
- Do NOT make up any factual database numbers, names of schemes, or core statistics.
- Do NOT mention any sources, document numbers (e.g. [1]), references, or URLs. Answer directly without citing references.
"""

    # 4. Read API key and invoke the Gemini LLM model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"response": "API Key is missing. Please configure GEMINI_API_KEY in the .env file."}), 500

    try:
        # Request the lightweight and fast gemini-2.5-flash-lite model
        answer = query_gemini(prompt, api_key, model="gemini-2.5-flash-lite")
    except Exception as e:
        print(f"Error invoking model: {e}")
        # Fallback to gemini-2.0-flash if needed
        try:
            answer = query_gemini(prompt, api_key, model="gemini-2.0-flash")
        except Exception as e2:
            return jsonify({"response": f"Error communicating with Gemini: {str(e2)}"}), 500

    # 5. Return the direct, grounded response to the client
    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(debug=True)
