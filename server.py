import os
import asyncio
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.google import GoogleModel

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
        raw = raw[start:end+1]

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

@app.route("/see-paper", methods=["POST"])
def see_paper():
    data = request.get_json()
    subject = data.get("subject", "Compulsory Social Studies")
    action = data.get("action", "generate")

    if action == "generate":
        prompt = f"""SEE (कक्षा १०) {subject} विषयको ७५ अंकको नमूना प्रश्नपत्र बनाउनुहोस्।
संरचना: १० छोटो प्रश्न (१ अंक), ८ मध्यम प्रश्न (४ अंक), ३ लामो प्रश्न (११ अंक)।
प्रश्नहरू नेपालीमा, SEE स्तरको। प्रत्येक प्रश्नको अंक देखाउनुहोस्।"""

        async def _gen():
            result = await agent.run(prompt)
            return result.output

        try:
            paper = asyncio.run(_gen())
            return jsonify({"paper": paper})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown action"}), 400

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
