import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow the HTML file to call this server

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.google import GoogleModel

SYSTEM_PROMPT = """You are an expert editor who detects AI-generated "slop" and rewrites it to sound human.

You will receive:
- The text
- The language (e.g. "English", "Nepali", "Hindi")

Your job:
1. Detect the AI-slop patterns: generic filler openings, robotic transitions, hollow intensifiers, listicle brain, emoji overload, no personal voice, perfect-but-empty.
2. Rewrite the text in the SAME language, keeping the same meaning but making it sound like a real person wrote it.
3. Keep it short. Do not add new facts. Do not change the message.
4. Return ONLY the rewritten text — no explanation, no preamble.

If the text is already human, return it unchanged.
"""

model = GoogleModel("gemini-flash-lite-latest")
agent = PydanticAgent(model, system_prompt=SYSTEM_PROMPT)

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "").strip()
    language = data.get("language", "English")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prompt = f"Language: {language}\n\nText:\n{text}"

    async def _ask():
        result = await agent.run(prompt)
        return result.output

    try:
        rewritten = asyncio.run(_ask())
        return jsonify({"rewritten": rewritten})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)