#!/usr/bin/env python3
"""
Image generator + gallery web app - generates images from a phone browser
using FLUX.1-schnell via Hugging Face, and saves every image + prompt to
Supabase so nothing gets lost.

Deployed on Render, works from your phone anytime, no computer needed.
"""

import os
import io
import base64
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from huggingface_hub import InferenceClient
from supabase import create_client

app = FastAPI()

# Set these in Render's environment variables (Dashboard -> Environment tab):
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # service_role key, backend-only
MODEL = "black-forest-labs/FLUX.1-schnell"
BUCKET = "thumbnails"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Static title-card template tool (canvas-based, runs entirely in the browser)
_template_path = os.path.join(os.path.dirname(__file__), "template_page.html")
with open(_template_path, "r", encoding="utf-8") as _f:
    TEMPLATE_PAGE = _f.read()

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thumbnail Generator</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 16px; }}
    textarea {{ width: 100%; height: 80px; font-size: 16px; padding: 8px; box-sizing: border-box; }}
    button, a.btn {{ display: block; width: 100%; padding: 14px; font-size: 16px; margin-top: 10px;
             background: #111; color: white; border: none; border-radius: 6px; text-align: center;
             text-decoration: none; box-sizing: border-box; }}
    a.btn.secondary {{ background: #555; }}
    img {{ width: 100%; margin-top: 16px; border-radius: 6px; display: block; }}
    .status {{ margin-top: 10px; color: #555; }}
    .history {{ margin-top: 40px; }}
    .history h3 {{ font-size: 15px; color: #555; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }}
    .grid figure {{ margin: 0; }}
    .grid img {{ margin-top: 0; aspect-ratio: 1; object-fit: cover; }}
    .grid figcaption {{ font-size: 11px; color: #777; margin-top: 2px; white-space: nowrap;
                overflow: hidden; text-overflow: ellipsis; }}
  </style>
</head>
<body>
  <h2>Thumbnail Generator</h2>
  <div style="margin-bottom:10px;"><a href="/template" style="color:#555; font-size:13px;">Title Card Template Tool &rarr;</a></div>
  <form action="/generate" method="post">
    <textarea name="prompt" placeholder="Describe the image..." required>{prompt}</textarea>
    <button type="submit">Generate</button>
  </form>
  {result}
  <div class="history">
    <h3>Recent generations</h3>
    <div class="grid">
      {history}
    </div>
  </div>
</body>
</html>
"""


def render_history(limit=12):
    if not supabase:
        return "<p class='status'>History disabled: SUPABASE_URL / SUPABASE_SERVICE_KEY not set on the server.</p>"
    try:
        rows = (
            supabase.table("thumbnails")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )
    except Exception as e:
        return f"<p class='status'>History error: {e}</p>"
    if not rows:
        return "<p class='status'>No saved generations yet.</p>"
    items = []
    for row in rows:
        public_url = supabase.storage.from_(BUCKET).get_public_url(row["image_path"])
        caption = row["prompt"][:40]
        items.append(f'<figure><a href="{public_url}" target="_blank"><img src="{public_url}"></a><figcaption>{caption}</figcaption></figure>')
    return "\n".join(items)


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.format(prompt="", result="", history=render_history())


@app.get("/generate")
def generate_redirect():
    return RedirectResponse(url="/")


@app.get("/template", response_class=HTMLResponse)
def template_tool():
    return TEMPLATE_PAGE


@app.post("/generate", response_class=HTMLResponse)
def generate(prompt: str = Form(...)):
    if not HF_TOKEN:
        return PAGE.format(prompt=prompt, result="<p class='status'>ERROR: HF_TOKEN not set on the server.</p>", history=render_history())

    try:
        client = InferenceClient(api_key=HF_TOKEN, provider="auto")
        image = client.text_to_image(prompt, model=MODEL)

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        # Show it immediately
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        result = f'<img src="data:image/png;base64,{b64}" /><a class="btn secondary" download="thumbnail.png" href="data:image/png;base64,{b64}">Download</a>'

        # Save to Supabase (storage + row), best-effort - don't fail the
        # whole request if this part has an issue
        if not supabase:
            result += "<p class='status'>(Not saved to history: SUPABASE_URL / SUPABASE_SERVICE_KEY not set)</p>"
        else:
            try:
                filename = f"{uuid.uuid4().hex}.png"
                supabase.storage.from_(BUCKET).upload(
                    filename, image_bytes, {"content-type": "image/png"}
                )
                supabase.table("thumbnails").insert({
                    "prompt": prompt,
                    "image_path": filename,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }).execute()
            except Exception as save_err:
                result += f"<p class='status'>(Saved image, but history save failed: {save_err})</p>"

        return PAGE.format(prompt=prompt, result=result, history=render_history())
    except Exception as e:
        return PAGE.format(prompt=prompt, result=f"<p class='status'>ERROR: {e}</p>", history=render_history())


@app.get("/health")
def health():
    return {"status": "ok"}
