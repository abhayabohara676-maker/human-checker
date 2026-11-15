import os
import asyncio
import json
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

Your job has TWO parts:

PART 1 — Detect the AI-slop patterns. Check for these, in the SAME language as the text:
- generic filler opening (e.g. "In today's fast-paced world...")
- robotic transitions (e.g. "Moreover", "Furthermore", "It is important to note")
- listicle brain (e.g. "Here are 5 ways to...")
- hollow intensifiers (e.g. "game-changer", "cutting-edge", "transformative")
- emoji overload (4+ emojis in a row)
- no personal voice (no "I", no opinion, no story)
- perfect but empty (grammatically flawless, no concrete details)

PART 2 — Rewrite the text in the SAME language, keeping the same meaning but making it sound like a real person wrote it. Match the requested tone exactly. Do not add new facts.

Return your answer as JSON ONLY, with this exact structure:
{
  "flags": ["list", "of", "slop", "patterns", "found"],
  "rewritten": "the rewritten text"
}

If no slop is found, use an empty flags list.
Return ONLY valid JSON. No explanation, no markdown, no preamble.
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
        raw = asyncio.run(_ask())
        # Try to parse the JSON the model returned
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
                cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
            flags = parsed.get("flags", [])
            rewritten = parsed.get("rewritten", "")
            if not rewritten:
                # fallback: if the model didn't follow the format, use raw as rewrite
                return jsonify({"flags": [], "rewritten": raw})
            return jsonify({"flags": flags, "rewritten": rewritten})
        except json.JSONDecodeError:
            # If the model didn't return valid JSON, fall back gracefully
            return jsonify({"flags": [], "rewritten": raw})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
