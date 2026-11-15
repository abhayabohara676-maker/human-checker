import os
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.google import GoogleModel

SYSTEM_PROMPT = """You are an expert editor who detects AI-generated "slop" and rewrites it to sound human.

You will receive:
- The text
- The language (e.g. "English", "Nepali", "Hindi")
- The tone (e.g. "Casual", "Professional", "Funny", "Short", "Bold")

Your job:
1. Detect the AI-slop patterns: generic filler openings, robotic transitions, hollow intensifiers, listicle brain, emoji overload, no personal voice, perfect-but-empty.
2. Rewrite the text in the SAME language, keeping the same meaning but making it sound like a real person wrote it.
3. Match the requested tone exactly. If tone is "Short", cut it to the essentials. If "Funny", add light humor. If "Professional", keep it clean and direct. If "Bold", make it punchy. If "Casual", make it relaxed.
4. Keep it short. Do not add new facts. Do not change the message.
5. Return ONLY the rewritten text — no explanation, no preamble.

If the text is already human, return it unchanged.
"""

model = GoogleModel("gemini-flash-lite-latest")
agent = PydanticAgent(model, system_prompt=SYSTEM_PROMPT)

@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "").strip()
    language = data.get("language", "English")
    tone = data.get("tone", "Casual")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    prompt = f"Language: {language}\nTone: {tone}\n\nText:\n{text}"

    async def _ask():
        result = await agent.run(prompt)
        return result.output

    try:
        rewritten = asyncio.run(_ask())
        return jsonify({"rewritten": rewritten})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
