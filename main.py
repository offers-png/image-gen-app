#!/usr/bin/env python3
"""
Image generator web app - generates images from a phone browser using
FLUX.1-schnell via Hugging Face Inference Providers.

Deployed on Render, this works from your phone anytime, no computer needed.
"""

import os
import io
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from huggingface_hub import InferenceClient

app = FastAPI()

# Set this in Render's environment variables (Dashboard -> Environment tab),
# NOT hardcoded here, since this code lives in a public/shared repo.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL = "black-forest-labs/FLUX.1-schnell"

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image Generator</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 16px; }}
    textarea {{ width: 100%; height: 80px; font-size: 16px; padding: 8px; box-sizing: border-box; }}
    button {{ width: 100%; padding: 14px; font-size: 16px; margin-top: 10px; background: #111; color: white; border: none; border-radius: 6px; }}
    img {{ width: 100%; margin-top: 16px; border-radius: 6px; }}
    .status {{ margin-top: 10px; color: #555; }}
  </style>
</head>
<body>
  <h2>Image Generator</h2>
  <form action="/generate" method="post">
    <textarea name="prompt" placeholder="Describe the image..." required>{prompt}</textarea>
    <button type="submit">Generate</button>
  </form>
  {result}
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.format(prompt="", result="")


@app.post("/generate", response_class=HTMLResponse)
def generate(prompt: str = Form(...)):
    if not HF_TOKEN:
        return PAGE.format(
            prompt=prompt,
            result="<p class='status'>ERROR: HF_TOKEN not set on the server.</p>",
        )
    try:
        client = InferenceClient(api_key=HF_TOKEN, provider="auto")
        image = client.text_to_image(prompt, model=MODEL)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        import base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result = f'<img src="data:image/png;base64,{b64}" />'
        return PAGE.format(prompt=prompt, result=result)
    except Exception as e:
        return PAGE.format(prompt=prompt, result=f"<p class='status'>ERROR: {e}</p>")


@app.get("/health")
def health():
    return {"status": "ok"}
