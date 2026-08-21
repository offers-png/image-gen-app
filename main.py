#!/usr/bin/env python3
"""
Image generator + gallery web app - generates images from a phone browser
using FLUX.1-schnell via Hugging Face, and saves every image + prompt to
Supabase so nothing gets lost.

Deployed on Render, works from your phone anytime, no computer needed.
"""

import os
import io
import json
import base64
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from gradio_client import Client
from supabase import create_client
from anthropic import Anthropic
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

# Set these in Render's environment variables (Dashboard -> Environment tab):
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # optional now, but keeps you signed-in for higher free quota
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")  # service_role key, backend-only
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SPACE = "black-forest-labs/FLUX.1-schnell"  # official free Space - no billing to your account
BUCKET = "thumbnails"

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_TITLE = os.path.join(FONT_DIR, "Anton-Regular.ttf")
FONT_BODY = os.path.join(FONT_DIR, "ArchivoBlack-Regular.ttf")


def safe_font(path, size):
    """Loads a TTF font, falling back to Pillow's built-in default if the
    file is missing - so a font upload issue degrades the look instead of
    breaking thumbnail generation entirely."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            listing = os.listdir(FONT_DIR)
        except Exception as list_err:
            listing = f"<could not list {FONT_DIR}: {list_err}>"
        print(f"WARNING: could not load font at {path}. "
              f"Contents of {FONT_DIR}: {listing}")
        return ImageFont.load_default(size=size)

# Same palettes as the client-side template tool, kept in sync by hand
PALETTES = {
    "dark":  {"bg1": (22, 22, 22),  "bg2": (10, 10, 10),  "accent": (180, 35, 26),  "text": (237, 234, 227), "stroke": (0, 0, 0)},
    "rust":  {"bg1": (58, 20, 8),   "bg2": (21, 5, 2),    "accent": (217, 83, 30),  "text": (237, 234, 227), "stroke": (0, 0, 0)},
    "steel": {"bg1": (35, 39, 42),  "bg2": (13, 15, 16),  "accent": (90, 169, 196), "text": (237, 234, 227), "stroke": (0, 0, 0)},
    "shock": {"bg1": (42, 26, 58),  "bg2": (18, 10, 28),  "accent": (242, 194, 48), "text": (242, 194, 48),  "stroke": (26, 15, 40)},
    "alert": {"bg1": (26, 46, 48),  "bg2": (10, 20, 21),  "accent": (62, 214, 214), "text": (255, 255, 255), "stroke": (122, 20, 20)},
    "bold":  {"bg1": (13, 36, 56),  "bg2": (6, 16, 24),   "accent": (242, 120, 47), "text": (255, 255, 255), "stroke": (10, 26, 42)},
}
VALID_TAGS = ["MONEY", "RELIGION", "POLITICS", "SYSTEM", "HUMANITY", "CULTURE"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Static title-card template tool (canvas-based, runs entirely in the browser)
_template_path = os.path.join(os.path.dirname(__file__), "template_page.html")
with open(_template_path, "r", encoding="utf-8") as _f:
    TEMPLATE_PAGE = _f.read()


def analyze_script(script_text: str) -> dict:
    """Ask Claude to read the script and pick a title, tag, color theme, and
    background image prompt for the thumbnail. Returns a plain dict."""
    system_prompt = (
        "You are a YouTube thumbnail strategist for a commentary channel called "
        "'Things That Really Matter' (tagline: 'The Truth. No Filter.'). Given a "
        "video script, output ONLY valid JSON (no markdown fences, no commentary) "
        "with exactly these keys:\n"
        '- "title": 3-5 punchy words, ALL CAPS, no ending punctuation, the strongest '
        "hook from the script (research shows 3-5 words performs best)\n"
        '- "tag": exactly one of MONEY, RELIGION, POLITICS, SYSTEM, HUMANITY, CULTURE '
        "- whichever fits the script's core topic best\n"
        '- "theme": exactly one of dark, rust, steel, shock, alert, bold - pick '
        "whichever color mood best matches the script's emotional tone (dark/rust = "
        "serious or angry, steel = cold/analytical, shock = urgent/shocking, alert = "
        "warning/danger, bold = confident/declarative)\n"
        '- "image_prompt": a single descriptive prompt (under 40 words) for an AI '
        "image generator to create the background. It must be symbolic or scenic "
        "ONLY - never depict specific real people, never depict any prophet or "
        "religious figure, and never include any text, letters, or writing in the "
        "image, since AI image models render text illegibly."
    )
    message = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": script_text[:8000]}],
    )

    # Join all text blocks (there should just be one, but be defensive)
    raw = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    ).strip()

    if not raw:
        raise ValueError(f"Empty response from model (stop_reason={message.stop_reason})")

    # Strip markdown code fences if the model wrapped the JSON despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: pull out the first {...} block in case there's stray
        # preamble/postamble text around the JSON
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find JSON in model response: {raw[:300]}")
        data = json.loads(match.group(0))

    # Guard against the model drifting outside allowed values
    if data.get("tag") not in VALID_TAGS:
        data["tag"] = "CULTURE"
    if data.get("theme") not in PALETTES:
        data["theme"] = "dark"
    if not data.get("title"):
        data["title"] = "YOUR TOPIC HERE"
    if not data.get("image_prompt"):
        data["image_prompt"] = "abstract dark symbolic background, cinematic, no text"
    return data


def generate_background_bytes(prompt: str) -> bytes:
    """Calls the free official FLUX.1-schnell Space and returns raw PNG bytes."""
    client = Client(SPACE, token=HF_TOKEN or None)
    space_result = client.predict(
        prompt, 0, True, 1024, 1024, 4, api_name="/infer",
    )
    image_path = space_result[0] if isinstance(space_result, (list, tuple)) else space_result
    with open(image_path, "rb") as f:
        return f.read()


def wrap_text(draw, text, font, max_width):
    words = text.split(" ")
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        if draw.textlength(test, font=font) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = test
    lines.append(line)
    return lines


def composite_thumbnail(background_bytes: bytes, title: str, tag: str, theme: str) -> bytes:
    """Builds the final 1280x720 thumbnail PNG, mirroring the client-side
    template tool's design: photo/gradient bg, accent bar, tag chip, outlined
    title text in the upper safe zone, bottom tagline strip."""
    W, H = 1280, 720
    p = PALETTES[theme]

    img = Image.new("RGB", (W, H), p["bg1"])
    draw = ImageDraw.Draw(img)

    if background_bytes:
        bg = Image.open(io.BytesIO(background_bytes)).convert("RGB")
        # cover-crop to fill 1280x720
        bg_ratio, box_ratio = bg.width / bg.height, W / H
        if bg_ratio > box_ratio:
            new_h = bg.height
            new_w = int(new_h * box_ratio)
            x = (bg.width - new_w) // 2
            bg = bg.crop((x, 0, x + new_w, new_h))
        else:
            new_w = bg.width
            new_h = int(new_w / box_ratio)
            y = (bg.height - new_h) // 2
            bg = bg.crop((0, y, new_w, y + new_h))
        bg = bg.resize((W, H))
        img.paste(bg, (0, 0))
        # dark overlay, heavier at top where the title sits
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for y in range(H):
            t = y / H
            alpha = int(165 - 90 * abs(t - 0.15) if t < 0.5 else 165 * t)
            alpha = max(60, min(190, alpha))
            odraw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        # accent color tint
        tint = Image.new("RGB", (W, H), p["accent"])
        img = Image.blend(img, tint, 0.10)
    else:
        # gradient background
        grad = Image.new("RGB", (W, H))
        gdraw = ImageDraw.Draw(grad)
        for x in range(W):
            t = x / W
            r = int(p["bg1"][0] + (p["bg2"][0] - p["bg1"][0]) * t)
            g = int(p["bg1"][1] + (p["bg2"][1] - p["bg1"][1]) * t)
            b = int(p["bg1"][2] + (p["bg2"][2] - p["bg1"][2]) * t)
            gdraw.line([(x, 0), (x, H)], fill=(r, g, b))
        img = grad

    draw = ImageDraw.Draw(img)

    # left accent bar
    draw.rectangle([0, 0, 18, H], fill=p["accent"])

    # tag chip
    tag_font = safe_font(FONT_BODY, 30)
    tag_pad_x = 22
    tag_w = draw.textlength(tag, font=tag_font) + tag_pad_x * 2
    draw.rectangle([80, 90, 80 + tag_w, 90 + 58], fill=p["accent"])
    draw.text((80 + tag_pad_x, 90 + 15), tag, font=tag_font, fill=(10, 10, 10))

    # show name small
    show_font = safe_font(FONT_BODY, 22)
    draw.text((80, 190), "THINGS THAT REALLY MATTER", font=show_font,
              fill=(237, 234, 227))

    # main title - outlined, upper safe zone
    title_font = safe_font(FONT_TITLE, 92)
    lines = wrap_text(draw, title.upper(), title_font, 1120)
    line_height = 100
    start_y = 400 - ((len(lines) - 1) * line_height) // 2
    stroke_w = 7
    for i, line in enumerate(lines):
        y = start_y + i * line_height
        draw.text((80, y), line, font=title_font, fill=p["text"],
                   stroke_width=stroke_w, stroke_fill=p["stroke"])

    # bottom strip
    draw.rectangle([0, 650, W, 720], fill=(0, 0, 0))
    strip_font = safe_font(FONT_BODY, 24)
    draw.text((80, 663), "THE TRUTH. NO FILTER.", font=strip_font, fill=p["accent"])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

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
    .dl-link {{ display: block; font-size: 11px; color: #111; text-decoration: underline;
                margin-top: 2px; }}
  </style>
</head>
<body>
  <h2>Thumbnail Generator</h2>
  <div style="margin-bottom:10px;">
    <a href="/template" style="color:#555; font-size:13px;">Title Card Template Tool &rarr;</a>
    &nbsp;|&nbsp;
    <a href="/auto" style="color:#555; font-size:13px;">Auto-Generate from Script &rarr;</a>
  </div>
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
        download_url = f"/download/{row['image_path']}"
        caption = row["prompt"][:40]
        items.append(
            f'<figure>'
            f'<a href="{public_url}" target="_blank"><img src="{public_url}"></a>'
            f'<figcaption>{caption}</figcaption>'
            f'<a href="{download_url}" class="dl-link">Download</a>'
            f'</figure>'
        )
    return "\n".join(items)


@app.get("/download/{filename}")
def download_image(filename: str):
    """Proxies the image through our own server with a Content-Disposition
    header, so browsers actually download it instead of just opening the
    cross-origin Supabase URL in a new tab."""
    if not supabase:
        return Response(content=b"History storage not configured.", status_code=503)
    try:
        data = supabase.storage.from_(BUCKET).download(filename)
    except Exception as e:
        return Response(content=f"Could not fetch image: {e}".encode(), status_code=404)
    return Response(
        content=data,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE.format(prompt="", result="", history=render_history())


@app.get("/generate")
def generate_redirect():
    return RedirectResponse(url="/")


@app.get("/template", response_class=HTMLResponse)
def template_tool():
    return TEMPLATE_PAGE


def save_to_history(prompt_label: str, image_bytes: bytes) -> str:
    """Uploads an image + row to Supabase. Returns a status message (empty
    string on success) so callers can surface failures without raising."""
    if not supabase:
        return "(Not saved to history: SUPABASE_URL / SUPABASE_SERVICE_KEY not set)"
    try:
        filename = f"{uuid.uuid4().hex}.png"
        supabase.storage.from_(BUCKET).upload(
            filename, image_bytes, {"content-type": "image/png"}
        )
        supabase.table("thumbnails").insert({
            "prompt": prompt_label,
            "image_path": filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return ""
    except Exception as save_err:
        return f"(Saved image, but history save failed: {save_err})"


@app.post("/generate", response_class=HTMLResponse)
def generate(prompt: str = Form(...)):
    try:
        image_bytes = generate_background_bytes(prompt)
    except Exception as e:
        return PAGE.format(prompt=prompt, result=f"<p class='status'>ERROR: {e}</p>", history=render_history())

    # Show it immediately
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    result = f'<img src="data:image/png;base64,{b64}" /><a class="btn secondary" download="thumbnail.png" href="data:image/png;base64,{b64}">Download</a>'

    save_msg = save_to_history(prompt, image_bytes)
    if save_msg:
        result += f"<p class='status'>{save_msg}</p>"

    return PAGE.format(prompt=prompt, result=result, history=render_history())


@app.get("/health")
def health():
    return {"status": "ok"}


AUTO_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Auto Thumbnail from Script</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 16px; }}
    textarea {{ width: 100%; height: 220px; font-size: 14px; padding: 8px; box-sizing: border-box; }}
    button {{ width: 100%; padding: 14px; font-size: 16px; margin-top: 10px;
             background: #111; color: white; border: none; border-radius: 6px; }}
    a.btn {{ display: block; width: 100%; padding: 14px; font-size: 16px; margin-top: 10px;
             background: #555; color: white; border: none; border-radius: 6px; text-align: center;
             text-decoration: none; box-sizing: border-box; }}
    img {{ width: 100%; margin-top: 16px; border-radius: 6px; display: block; }}
    .status {{ margin-top: 10px; color: #555; }}
    .meta {{ margin-top: 10px; font-size: 13px; color: #555; background: #f4f4f4; padding: 10px; border-radius: 6px; }}
    .meta b {{ color: #111; }}
  </style>
</head>
<body>
  <h2>Auto-Generate from Script</h2>
  <div style="margin-bottom:10px;"><a href="/" style="color:#555; font-size:13px;">&larr; AI Image Generator</a></div>
  <p style="font-size:13px; color:#666;">Paste your full voiceover script below. Claude reads it and picks the title, topic tag, color theme, and background image - then the full thumbnail gets built automatically.</p>
  <form action="/auto" method="post">
    <textarea name="script" placeholder="Paste your script here..." required>{script}</textarea>
    <button type="submit">Generate Thumbnail</button>
  </form>
  {result}
</body>
</html>
"""


@app.get("/auto", response_class=HTMLResponse)
def auto_home():
    return AUTO_PAGE.format(script="", result="")


@app.post("/auto", response_class=HTMLResponse)
def auto_generate(script: str = Form(...)):
    if not anthropic_client:
        return AUTO_PAGE.format(
            script=script,
            result="<p class='status'>ERROR: ANTHROPIC_API_KEY not set on the server.</p>",
        )

    try:
        analysis = analyze_script(script)
    except Exception as e:
        return AUTO_PAGE.format(script=script, result=f"<p class='status'>ERROR analyzing script: {e}</p>")

    title = analysis.get("title", "YOUR TOPIC HERE")
    tag = analysis.get("tag", "CULTURE")
    theme = analysis.get("theme", "dark")
    image_prompt = analysis.get("image_prompt", "")

    try:
        bg_bytes = generate_background_bytes(image_prompt)
    except Exception as e:
        bg_bytes = None
        bg_error = f"<p class='status'>Background generation failed ({e}), using solid color instead.</p>"
    else:
        bg_error = ""

    try:
        final_bytes = composite_thumbnail(bg_bytes, title, tag, theme)
    except Exception as e:
        return AUTO_PAGE.format(script=script, result=f"<p class='status'>ERROR compositing thumbnail: {e}</p>")

    b64 = base64.b64encode(final_bytes).decode("utf-8")
    meta = (
        f"<div class='meta'><b>Title:</b> {title}<br>"
        f"<b>Tag:</b> {tag}<br>"
        f"<b>Theme:</b> {theme}<br>"
        f"<b>Image prompt used:</b> {image_prompt}</div>"
    )
    result = (
        f'<img src="data:image/png;base64,{b64}" />'
        f'<a class="btn" download="ttrm-{tag.lower()}.png" href="data:image/png;base64,{b64}">Download</a>'
        + bg_error + meta
    )

    save_msg = save_to_history(f"[AUTO] {title}", final_bytes)
    if save_msg:
        result += f"<p class='status'>{save_msg}</p>"

    return AUTO_PAGE.format(script=script, result=result)
