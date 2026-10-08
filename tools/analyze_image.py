#!/usr/bin/env python3
"""
External vision worker for GitHub Actions.

Images placed in vision/inbox/ are sent to the configured Hugging Face
vision Space. The worker writes only text/JSON results back to the repo.
GPT is not involved in the image-analysis step.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from gradio_client import Client, handle_file

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def choose_endpoint(client):
    """Find a likely multimodal generation endpoint from the Space API."""
    api = client.view_api(return_format="dict")
    named = api.get("named_endpoints", {}) if isinstance(api, dict) else {}

    preferred = ("/generate", "/chat", "/predict", "/run", "/inference")
    for name in preferred:
        if name in named:
            return name, named[name]

    for name, spec in named.items():
        text = (name + " " + json.dumps(spec)).lower()
        if any(k in text for k in ("image", "multimodal", "vision", "chat")):
            return name, spec

    if len(named) == 1:
        name = next(iter(named))
        return name, named[name]

    raise RuntimeError(
        "Could not identify a vision endpoint. Available endpoints: "
        + ", ".join(named.keys())
    )


def build_args(spec, image_path, prompt):
    """Build arguments for common Gradio image+text endpoint signatures."""
    params = spec.get("parameters", []) if isinstance(spec, dict) else []
    args = []

    for p in params:
        name = str(p.get("parameter_name", p.get("name", ""))).lower()
        typ = str(p.get("type", "")).lower()

        if "image" in name or typ in {"image", "filepath"}:
            args.append(handle_file(str(image_path)))
        elif any(k in name for k in ("prompt", "question", "query", "instruction", "text", "message")):
            args.append(prompt)
        elif "system" in name:
            args.append("")
        elif "history" in name or "messages" in name:
            args.append([])
        elif "temperature" in name:
            args.append(0.2)
        elif "max" in name and "token" in name:
            args.append(1024)
        elif "stream" in name:
            args.append(False)
        else:
            default = p.get("default")
            args.append(default)

    return args


def analyze(image_path, client, endpoint, spec):
    prompt = os.environ.get(
        "VISION_PROMPT",
        "Describe this image accurately and conservatively. "
        "Identify visible objects, people, text, layout, colors, and notable details. "
        "Do not guess facts that cannot be visually supported."
    )
    args = build_args(spec, image_path, prompt)
    result = client.predict(*args, api_name=endpoint)
    return result


def main():
    inbox = Path(os.environ.get("VISION_INBOX", "vision/inbox"))
    results = Path(os.environ.get("VISION_RESULTS", "vision/results"))
    results.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in inbox.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print("No images found in", inbox)
        return 0

    space = os.environ.get("HF_VISION_SPACE", "akhaliq/Qwen3-VL-2B-Instruct")
    token = os.environ.get("HF_TOKEN") or None
    client = Client(space, hf_token=token)

    endpoint, spec = choose_endpoint(client)
    print(f"Using vision Space: {space}")
    print(f"Using endpoint: {endpoint}")

    for image_path in images:
        started = datetime.now(timezone.utc)
        try:
            raw = analyze(image_path, client, endpoint, spec)
            record = {
                "image": str(image_path.relative_to(inbox)),
                "model_space": space,
                "endpoint": endpoint,
                "analyzed_at": started.isoformat(),
                "result": raw,
            }
            out = results / (image_path.stem + ".json")
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            print("Wrote", out)
        except Exception as exc:
            out = results / (image_path.stem + ".error.json")
            out.write_text(json.dumps({
                "image": str(image_path.relative_to(inbox)),
                "model_space": space,
                "endpoint": endpoint,
                "analyzed_at": started.isoformat(),
                "error": repr(exc),
            }, indent=2) + "\n", encoding="utf-8")
            print(f"ERROR analyzing {image_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
