#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from gradio_client import Client

SCHEMA_HINT = {
    "image": {"source_url": "", "filename": "", "format": "", "width": 0, "height": 0},
    "visual_summary": "",
    "ocr": [],
    "layout": {"description": "", "regions": []},
    "visual_elements": [],
    "relationships": [],
    "colors": [],
    "typography": [],
    "semantic_interpretation": "",
    "uncertainties": [],
    "confidence": 0.0
}

def raw_github_url(url):
    p = urlparse(url)
    if p.netloc.lower() == "github.com":
        parts = p.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, ref = parts[:4]
            path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    return url

def download_image(url, out):
    r = requests.get(raw_github_url(url), timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    with Image.open(out) as im:
        im.verify()

def extract_json(text):
    text = text.strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise

def call_space(image_path, prompt):
    space = os.environ.get("HF_SPACE", "akhaliq/Qwen3-VL-2B-Instruct")
    token = os.environ.get("HF_TOKEN") or None
    client = Client(space, token=token)
    message = {"text": prompt, "files": [str(image_path)]}
    last_error = None
    for endpoint in ("/chat", "/qwen_chat_fn"):
        try:
            result = client.predict(message, [], api_name=endpoint)
            if isinstance(result, tuple):
                result = result[0]
            return str(result)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Vision Space call failed: {last_error}")

def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: analyze_image.py request.json output.json")
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    source_url = request["image_url"]
    user_prompt = request.get("prompt", "").strip()
    prompt = f"""You are an image-analysis engine. Analyze the attached image and return ONLY valid JSON.
Do not use markdown fences. Do not invent text that cannot be read.

Return an object matching this shape:
{json.dumps(SCHEMA_HINT, ensure_ascii=False, indent=2)}

Rules:
- OCR must preserve visible text as exactly as possible.
- Describe spatial/layout relationships explicitly.
- For unreadable text, use an uncertainty note rather than guessing.
- Separate observation from interpretation.
- Keep confidence between 0 and 1.
- Output JSON only.

User task:
{user_prompt or 'Provide a detailed visual inventory, OCR, layout analysis, and semantic interpretation.'}"""
    work = Path(".analysis_work")
    work.mkdir(exist_ok=True)
    image_path = work / "input_image"
    download_image(source_url, image_path)
    with Image.open(image_path) as im:
        width, height = im.size
        fmt = im.format or ""
    parsed = extract_json(call_space(image_path, prompt))
    if not isinstance(parsed, dict):
        raise ValueError("Model did not return a JSON object.")
    parsed.setdefault("image", {})
    parsed["image"].update({
        "source_url": source_url,
        "filename": Path(urlparse(source_url).path).name,
        "format": fmt,
        "width": width,
        "height": height,
    })
    Path(sys.argv[2]).write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
