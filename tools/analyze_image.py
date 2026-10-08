#!/usr/bin/env python3
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
MODEL_ID = os.environ.get("VISION_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")

PROMPT = os.environ.get("VISION_PROMPT", """Act as a meticulous visual-analysis engine. Create a detailed machine-useful representation of everything visually relevant in the image so another AI can reason about it later without seeing the pixels. Return JSON with these top-level fields: image_summary, exact_text, subjects, objects, spatial_relationships, appearance, pose_and_geometry, composition, colors_and_lighting, measurements_and_data, semantic_relationships, notable_details, uncertainties, and visual_edit_sensitive_details. Preserve visible text, headings, labels, numbers, units, locations, relative size, attributes, relationships, and confidence. For charts, diagrams, tables, or infographics, extract their structure and readable data. Record subtle/edit-sensitive details such as small objects, shadows, borders, markings, background elements, overlaps, and asymmetries. Distinguish observation from interpretation. Do not invent hidden facts. If something is uncertain or unreadable, say so explicitly.""")

def now(): return datetime.now(timezone.utc).isoformat()
def sp(results, image): return results / (image.stem + ".status.json")
def status(results, image, inbox, state, **extra):
    p = {"image": str(image.relative_to(inbox)), "status": state, "updated_at": now(), **extra}
    sp(results, image).write_text(json.dumps(p, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def normalize(x):
    if isinstance(x, (list, tuple)) and len(x) == 1: x = x[0]
    if isinstance(x, str):
        t = x.strip()
        if t.startswith("```"):
            lines = t.splitlines()[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            t = "\n".join(lines).strip()
            if t.lower().startswith("json"): t = t[4:].lstrip()
        try: return json.loads(t)
        except Exception: return x
    return x

def analyze(image, processor, model):
    pic = Image.open(image).convert("RGB")
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, images=[pic], return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=int(os.environ.get("VISION_MAX_NEW_TOKENS","2048")), do_sample=False)
    n = inputs["input_ids"].shape[1]
    return processor.batch_decode(ids[:, n:], skip_special_tokens=True)[0].strip()

def main():
    inbox, results = Path("vision/inbox"), Path("vision/results")
    results.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in inbox.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    pending = []
    for image in images:
        f = sp(results, image)
        old = {}
        if f.exists():
            try: old = json.loads(f.read_text(encoding="utf-8"))
            except Exception: pass
        if old.get("status") == "COMPLETE" or (old.get("status") == "ERROR" and os.environ.get("FORCE_RETRY_ERRORS","").lower() not in {"1","true","yes"}):
            continue
        pending.append(image)
    if not pending: print("No pending images."); return 0
    for image in pending: status(results, image, inbox, "PROCESSING", model=MODEL_ID, runtime="github-hosted-runner-cpu")
    try:
        print("Loading", MODEL_ID)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cpu").eval()
    except Exception as e:
        for image in pending: status(results, image, inbox, "ERROR", model=MODEL_ID, runtime="github-hosted-runner-cpu", error=repr(e))
        return 0
    for image in pending:
        started = now()
        try:
            raw = analyze(image, processor, model)
            out = results / (image.stem + ".json")
            rec = {"image":str(image.relative_to(inbox)), "model":MODEL_ID, "runtime":"github-hosted-runner-cpu", "analysis_type":"structured_visual_representation", "schema_version":"3.0", "analyzed_at":started, "completed_at":now(), "result":normalize(raw)}
            out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            status(results, image, inbox, "COMPLETE", model=MODEL_ID, runtime="github-hosted-runner-cpu", result_file=str(out), completed_at=rec["completed_at"])
            print("Completed", image)
        except Exception as e:
            status(results, image, inbox, "ERROR", model=MODEL_ID, runtime="github-hosted-runner-cpu", error=repr(e))
            print("ERROR", image, e, file=sys.stderr)
    return 0
if __name__ == "__main__": raise SystemExit(main())