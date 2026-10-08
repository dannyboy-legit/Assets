#!/usr/bin/env python3
"""
External vision worker for GitHub Actions.

Images placed in vision/inbox/ are sent to the configured Hugging Face
vision Space. The worker writes text/JSON results back to the repo.
GPT is not involved in the image-analysis step.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from gradio_client import Client, handle_file

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


def now():
    return datetime.now(timezone.utc).isoformat()


def status_path(results, image_path, inbox):
    return results / (image_path.stem + ".status.json")


def write_status(results, image_path, inbox, status, **extra):
    payload = {
        "image": str(image_path.relative_to(inbox)),
        "status": status,
        "updated_at": now(),
        **extra,
    }
    status_path(results, image_path, inbox).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def choose_endpoint(client):
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


def build_args(spec, image_path, prompt, endpoint_name_hint=""):
    params = spec.get("parameters", []) if isinstance(spec, dict) else []
    args = []

    for p in params:
        name = str(p.get("parameter_name", p.get("name", ""))).lower()
        typ = str(p.get("type", "")).lower()

        if "message" in name and (typ in {"dict", "object"} or "chat" in endpoint_name_hint):
            args.append({
                "text": prompt,
                "files": [handle_file(str(image_path))],
            })
        elif "image" in name or typ in {"image", "filepath"}:
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
            args.append(4096)
        elif "stream" in name:
            args.append(False)
        else:
            args.append(p.get("default"))

    return args


def normalize_result(raw):
    """Turn JSON-looking model output into real structured JSON when possible."""
    value = raw
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].lstrip()
        try:
            return json.loads(text)
        except Exception:
            return value
    return value

def analyze(image_path, client, endpoint, spec):
    prompt = os.environ.get(
        "VISION_PROMPT",
        """Act as a meticulous visual-analysis engine. Your job is NOT to write a short caption. "
        "Create a detailed, machine-useful representation of everything visually relevant in the image, "
        "so another AI can reason about the image later without seeing the pixels. "
        "Return JSON with these top-level fields: "
        "image_summary, exact_text, subjects, objects, spatial_relationships, appearance, pose_and_geometry, "
        "composition, colors_and_lighting, measurements_and_data, semantic_relationships, notable_details, "
        "uncertainties, and visual_edit_sensitive_details. "
        "For every important item, preserve location (left/right/top/bottom/center, foreground/background), "
        "relative size, attributes, relationships, and confidence when useful. "
        "Transcribe visible text as accurately as possible and preserve headings, labels, numbers, and units. "
        "For charts, diagrams, tables, or infographics, extract their structure and readable data rather than "
        "merely describing them. Describe people and objects at a level useful for detecting later edits: "
        "clothing, colors, pose, orientation, proportions, facial/hair details when visible, and interactions. "
        "Record subtle or potentially editable details such as small objects, shadows, borders, markings, "
        "background elements, overlaps, and asymmetries. Distinguish direct visual observations from interpretation. "
        "Do not invent hidden facts. If something is uncertain or unreadable, say so explicitly. "
        "Prefer comprehensive structured observations over elegant prose."""
    )
    return client.predict(
        *build_args(spec, image_path, prompt, endpoint_name_hint=endpoint),
        api_name=endpoint,
    )


def main():
    inbox = Path(os.environ.get("VISION_INBOX", "vision/inbox"))
    results = Path(os.environ.get("VISION_RESULTS", "vision/results"))
    results.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        p for p in inbox.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    # Avoid wasting external vision quota on images that already completed.
    # ERROR images are not retried automatically unless FORCE_RETRY_ERRORS=true.
    force_retry_errors = os.environ.get("FORCE_RETRY_ERRORS", "").lower() in {"1", "true", "yes"}
    images = []
    for image_path in all_images:
        status_file = status_path(results, image_path, inbox)
        if status_file.exists():
            try:
                previous = json.loads(status_file.read_text(encoding="utf-8"))
                previous_status = previous.get("status")
            except Exception:
                previous_status = None
            if previous_status == "COMPLETE":
                print("Skipping already-complete image:", image_path)
                continue
            if previous_status == "ERROR" and not force_retry_errors:
                print("Skipping previous error; set FORCE_RETRY_ERRORS=true to retry:", image_path)
                continue
        images.append(image_path)

    if not images:
        print("No images found in", inbox)
        return 0

    space = os.environ.get("HF_VISION_SPACE", "akhaliq/Qwen3-VL-2B-Instruct")
    token = os.environ.get("HF_TOKEN") or None

    for image_path in images:
        write_status(
            results, image_path, inbox, "PROCESSING",
            model_space=space,
            started_at=now(),
        )

    try:
        client_kwargs = {}
        if token:
            client_kwargs["token"] = token
        client = Client(space, **client_kwargs)
        endpoint, spec = choose_endpoint(client)
        print(f"Using vision Space: {space}")
        print(f"Using endpoint: {endpoint}")
    except Exception as exc:
        for image_path in images:
            write_status(
                results, image_path, inbox, "ERROR",
                model_space=space,
                error=repr(exc),
            )
        return 0

    for image_path in images:
        started = now()
        try:
            raw = None
            last_error = None
            for attempt in range(1, 4):
                try:
                    raw = analyze(image_path, client, endpoint, spec)
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        print(f"Attempt {attempt} failed for {image_path}; retrying...")
                        time.sleep(10 * attempt)
            if raw is None:
                raise last_error

            record = {
                "image": str(image_path.relative_to(inbox)),
                "model_space": space,
                "analysis_type": "structured_visual_representation",
                "schema_version": "2.0",
                "endpoint": endpoint,
                "analyzed_at": started,
                "completed_at": now(),
                "result": normalize_result(raw),
            }
            out = results / (image_path.stem + ".json")
            out.write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            write_status(
                results, image_path, inbox, "COMPLETE",
                model_space=space,
                endpoint=endpoint,
                completed_at=record["completed_at"],
                result_file=str(out),
            )
            print("Completed", image_path)
        except Exception as exc:
            error_file = results / (image_path.stem + ".error.json")
            error_file.write_text(
                json.dumps({
                    "image": str(image_path.relative_to(inbox)),
                    "model_space": space,
                    "endpoint": endpoint,
                    "analyzed_at": started,
                    "error": repr(exc),
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            write_status(
                results, image_path, inbox, "ERROR",
                model_space=space,
                endpoint=endpoint,
                error=repr(exc),
            )
            print(f"ERROR analyzing {image_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
