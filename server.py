import os
import asyncio
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)                                   # gunicorn looks for this name: server:app
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024      # allow photo uploads up to 5 MB
CORS(app)

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.google import GoogleModel

from see_paper import bp as see_bp                      # SEE paper + photo check
app.register_blueprint(see_bp)

# ---------------- human checker (your old tool) ----------------

SYSTEM_PROMPT = """You are an expert editor. You MUST return valid JSON only.

Analyze the text for these AI-slop patterns:
- generic filler opening
- robotic transitions
- listicle brain
- hollow intensifiers
- emoji overload
- no personal voice
- perfect but empty

Then rewrite the text to sound human, in the same language, matching the requested tone.

Return EXACTLY this JSON structure, nothing else:
{"flags": ["pattern1", "pattern2"], "rewritten": "the rewritten text"}

Do not use markdown. Do not add explanation. Only the JSON object.
"""

model = GoogleModel("gemini-flash-lite-latest")
agent = PydanticAgent(model, system_prompt=SYSTEM_PROMPT)

MAX_CHUNK = 1500


def split_into_chunks(text: str):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= MAX_CHUNK:
            current = (current + "\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            if len(para) > MAX_CHUNK:
                sentences = para.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|")
                cur2 = ""
                for s in sentences:
                    if len(cur2) + len(s) + 1 <= MAX_CHUNK:
                        cur2 = (cur2 + " " + s).strip()
                    else:
                        if cur2:
                            chunks.append(cur2)
                        cur2 = s.strip()
                if cur2:
                    chunks.append(cur2)
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


async def rewrite_chunk(text: str, language: str, tone: str):
    prompt = f"Language: {language}\nTone: {tone}\n\nText:\n{text}"
    result = await agent.run(prompt)
    raw = result.output.strip()

    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    try:
        parsed = json.loads(raw)
        return parsed.get("flags", []), parsed.get("rewritten", text)
    except Exception:
        return [], text


@app.route("/rewrite", methods=["POST"])
def rewrite():
    data = request.get_json()
    text = data.get("text", "").strip()
    language = data.get("language", "English")
    tone = data.get("tone", "Casual")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    chunks = split_into_chunks(text)
    all_flags = set()
    rewritten_parts = []

    async def _process():
        for chunk in chunks:
            flags, rewritten = await rewrite_chunk(chunk, language, tone)
            for f in flags:
                all_flags.add(f)
            rewritten_parts.append(rewritten)

    try:
        asyncio.run(_process())
        return jsonify({
            "flags": list(all_flags),
            "rewritten": "\n\n".join(rewritten_parts),
            "chunks": len(chunks)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------- pages ----------------

@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/human_checker.html")
def checker():
    return send_from_directory(".", "human_checker.html")


@app.route("/see_test.html")
def see_test():
    return send_from_directory(".", "see_test.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
