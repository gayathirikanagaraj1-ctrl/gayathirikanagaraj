"""ComicCraft - AI comic generator. Flask + Gemini (story) + image backend + PDF export."""
import base64, io, json, os, re, textwrap, time, uuid
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True, encoding="utf-8-sig")
app = Flask(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
FALLBACK_MODEL = "gemini-flash-latest"
BACKUP_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
SETTINGS = ["School", "Forest", "Space", "City"]
TONES = ["Light-hearted", "Dramatic", "Poetic", "Funny"]
STYLES = ["Anime", "Pixel Art", "Comic Book", "Realistic"]
COMICS = {}  # comic_id -> comic dict (last 20 kept in memory)


def env(name, default=""):
    return (os.getenv(name) or default).strip().strip('"').strip("'")


# ---------- Gemini ----------
def models_to_try():
    seen, out = set(), []
    for m in [env("GEMINI_FLASH_MODEL", FALLBACK_MODEL), FALLBACK_MODEL] + BACKUP_MODELS:
        if m and m not in seen:
            seen.add(m); out.append(m)
    return out


def friendly_error(status, body):
    if status in (400, 401, 403):
        return "API key problem. Get a key starting with 'AIza' from aistudio.google.com/apikey and put it in .env"
    if status == 404:
        return "Model name not found. Set GEMINI_FLASH_MODEL=gemini-flash-latest in .env"
    if status in (500, 502, 503, 504):
        return "Google servers are busy right now (temporary). Wait 30 seconds and click Generate again."
    if status == 429:
        return "Quota/rate limit hit. Wait a minute and try again (Pro models have no free tier)."
    return f"Gemini error {status}: {body[:150]}"


def call_gemini(prompt):
    key = env("GEMINI_API_KEY")
    if not key or key.endswith("your_key_here"):
        raise RuntimeError("NO_KEY")
    last = "Unknown error"
    for model in models_to_try():
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{API_ROOT}/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}],
                          "generationConfig": {"responseMimeType": "application/json", "temperature": 0.9}},
                    timeout=90)
            except requests.RequestException as e:
                last = f"Network error: {e}"
                break
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts)
                except (KeyError, IndexError, ValueError):
                    last = "Gemini returned an empty response. Try a different idea."
                    break
            last = friendly_error(r.status_code, r.text)
            if r.status_code in (400, 401, 403):
                raise RuntimeError(last)
            if r.status_code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            break
    raise RuntimeError(last)


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def build_prompt(f, panels):
    return textwrap.dedent(f"""
    You are a comic book writer. Write a {panels}-panel comic.
    Story idea: {f['prompt']}
    Main character: {f['character']}
    Setting: {f['setting']}
    Tone: {f['tone']}
    Art style: {f['art_style']}
    Return ONLY valid JSON in exactly this shape:
    {{"title": "comic title",
      "panels": [{{"title": "short panel title",
                   "scene_description": "2 sentences describing the scene",
                   "caption": "short ambient line, max 20 words",
                   "narration": "2 sentences about the character's actions or feelings",
                   "image_prompt": "English visual prompt for an image generator: {f['art_style']} style, {f['character']} in {f['setting']}, lighting, mood"}}]}}
    Exactly {panels} panels with a clear beginning, middle and end.
    """).strip()


def clean_story(data, panels, f):
    items = data.get("panels") if isinstance(data, dict) else None
    if not items:
        raise ValueError("no panels")
    out = []
    for i, p in enumerate(items[:panels], 1):
        scene = str(p.get("scene_description") or p.get("scene") or "").strip()
        out.append({"number": i, "title": str(p.get("title") or f"Scene {i}").strip(),
                    "scene": scene, "caption": str(p.get("caption", "")).strip(),
                    "narration": str(p.get("narration", "")).strip(),
                    "image_prompt": str(p.get("image_prompt") or scene or f["prompt"]).strip()})
    return {"title": str(data.get("title") or "My Comic").strip(), "panels": out}


def demo_story(f, panels):
    c, s = f["character"], f["setting"]
    beats = [("The Beginning", f"{c} arrives in the {s.lower()}.", "A quiet moment before the adventure."),
             ("Into the Unknown", f"{c} explores deeper, curious and careful.", "Something feels different here."),
             ("The Challenge", f"A big problem appears and {c} must decide what to do.", "The air grows tense."),
             ("The Big Moment", f"{c} acts bravely and everything changes.", "Courage lights the way."),
             ("Resolution", f"{c} looks back, proud of the journey.", "Every ending is a new start."),
             ("Epilogue", f"A new story waits for {c}.", "To be continued...")]
    out = []
    for i in range(panels):
        t, n, cap = beats[i % len(beats)]
        out.append({"number": i + 1, "title": t, "scene": f"{n} ({f['tone']} tone, {f['art_style']} style).",
                    "caption": cap, "narration": n, "image_prompt": f"{f['art_style']} style, {c} in {s}, {t}"})
    return {"title": "Demo Comic", "panels": out}


# ---------- Images ----------
def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf",
              "/Library/Fonts/Arial Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


PAL = [((255, 209, 102), (239, 71, 111)), ((6, 214, 160), (17, 138, 178)), ((247, 140, 107), (7, 59, 76)),
       ((199, 125, 255), (58, 12, 164)), ((255, 175, 204), (94, 96, 206)), ((144, 224, 239), (2, 62, 138))]


def placeholder_image(title, idx):
    w = h = 640
    c1, c2 = PAL[idx % len(PAL)]
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)))
    d.ellipse([w - 190, 50, w - 80, 160], fill=(255, 255, 255))
    d.polygon([(0, 470), (170, 270), (320, 450), (450, 300), (w, 480), (w, h), (0, h)], fill=(30, 40, 60))
    d.polygon([(0, 540), (220, 400), (400, 530), (w, 420), (w, h), (0, h)], fill=(15, 20, 35))
    f = _font(34)
    y = 70
    for line in textwrap.wrap(title or "Comic panel", 20)[:3]:
        d.text((40, y), line, font=f, fill=(255, 255, 255))
        y += 44
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def to_jpeg(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.width < 64 or img.height < 64:
        raise ValueError("tiny image")
    img.thumbnail((768, 768))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _pollinations(prompt, idx):
    # Free service allows ~1 request at a time: retry politely if it says "busy"
    for attempt in range(4):
        try:
            r = requests.get(f"https://image.pollinations.ai/prompt/{quote(prompt[:400])}",
                             params={"width": 768, "height": 768, "nologo": "true", "seed": idx + 1}, timeout=60)
            if r.status_code == 200:
                return to_jpeg(r.content)
        except Exception:
            pass
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("pollinations busy")


def _hf(prompt):
    r = requests.post(f"https://router.huggingface.co/hf-inference/models/{env('HF_MODEL', 'stabilityai/stable-diffusion-xl-base-1.0')}",
                      headers={"Authorization": f"Bearer {env('HF_TOKEN')}"}, json={"inputs": prompt}, timeout=90)
    if r.status_code != 200:
        raise RuntimeError("hf failed")
    return to_jpeg(r.content)


def fetch_image(prompt, idx, title):
    backend = env("IMAGE_BACKEND", "pollinations").lower()
    chain = []
    if backend == "pollinations":
        chain.append(("pollinations", lambda: _pollinations(prompt, idx)))
    if (backend == "hf_api" or backend == "pollinations") and env("HF_TOKEN"):
        chain.append(("hf_api", lambda: _hf(prompt)))
    for name, fn in chain:  # try each real backend, then the drawn placeholder
        try:
            return fn(), name
        except Exception:
            continue
    return placeholder_image(title, idx), "placeholder"


# ---------- Comic build / PDF ----------
def make_comic(f, panels=4):
    notice, demo = "", False
    try:
        story = clean_story(parse_json(call_gemini(build_prompt(f, panels))), panels, f)
    except RuntimeError as e:
        if str(e) != "NO_KEY":
            raise
        story, demo = demo_story(f, panels), True
        has_env = os.path.exists(os.path.join(BASE_DIR, ".env"))
        notice = (".env found" if has_env else ".env NOT found next to app.py") + ", but GEMINI_API_KEY is empty. Showing a demo story."
    except (ValueError, KeyError, TypeError, AttributeError):
        raise RuntimeError("Gemini sent a badly formatted story. Please click Generate again.")
    # one image at a time for free backends (parallel requests get rejected, so only 1 panel got an image)
    workers = 4 if env("IMAGE_BACKEND", "pollinations").lower() == "placeholder" else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(lambda p: fetch_image(p["image_prompt"], p["number"] - 1, p["title"]), story["panels"]))
    for p, (data, src) in zip(story["panels"], res):
        p["image_bytes"], p["image_source"] = data, src
        p["image_b64"] = base64.b64encode(data).decode()
    story.update(id=uuid.uuid4().hex[:12], demo=demo, notice=notice, form=f)
    COMICS[story["id"]] = story
    while len(COMICS) > 20:
        COMICS.pop(next(iter(COMICS)))
    return story


def build_pdf(comic):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer
    reg, bold, ital = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    try:
        d = "/usr/share/fonts/truetype/dejavu/"
        pdfmetrics.registerFont(TTFont("DJ", d + "DejaVuSans.ttf"))
        pdfmetrics.registerFont(TTFont("DJB", d + "DejaVuSans-Bold.ttf"))
        pdfmetrics.registerFont(TTFont("DJI", d + "DejaVuSans-Oblique.ttf"))
        reg, bold, ital = "DJ", "DJB", "DJI"
    except Exception:
        pass
    st_t = ParagraphStyle("t", fontName=bold, fontSize=22, alignment=1, spaceAfter=14)
    st_h = ParagraphStyle("h", fontName=bold, fontSize=14, spaceAfter=6)
    st_i = ParagraphStyle("i", fontName=ital, fontSize=10, textColor=colors.HexColor("#444444"), spaceBefore=6, leading=13)
    st_b = ParagraphStyle("b", fontName=reg, fontSize=10, spaceBefore=4, leading=13)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title=comic["title"])
    story = [Paragraph(escape(comic["title"]), st_t)]
    for p in comic["panels"]:
        img = RLImage(io.BytesIO(p["image_bytes"]), width=11 * cm, height=11 * cm)
        story.append(KeepTogether([
            Paragraph(f"Panel {p['number']}: {escape(p['title'])}", st_h), img,
            Paragraph(escape(p["scene"]), st_i),
            Paragraph(f"<b>Caption:</b> {escape(p['caption'])}", st_b),
            Paragraph(f"<b>Narration:</b> {escape(p['narration'])}", st_b), Spacer(1, 16)]))
    doc.build(story)
    buf.seek(0)
    return buf


# ---------- Routes ----------
def read_form(src):
    g = lambda k, d: str(src.get(k) or d).strip()
    f = {"prompt": g("prompt", "")[:500], "character": g("character", "Hero")[:40],
         "setting": g("setting", "Forest"), "tone": g("tone", "Dramatic"), "art_style": g("art_style", "Comic Book")}
    f["setting"] = f["setting"] if f["setting"] in SETTINGS else "Forest"
    f["tone"] = f["tone"] if f["tone"] in TONES else "Dramatic"
    f["art_style"] = f["art_style"] if f["art_style"] in STYLES else "Comic Book"
    try:
        panels = max(2, min(int(src.get("panels", 4)), 8))
    except (TypeError, ValueError):
        panels = 4
    return f, panels


@app.get("/")
def home():
    return render_template("index.html", settings=SETTINGS, tones=TONES, styles=STYLES, form={}, error=None)


@app.post("/generate")
def generate():
    f, panels = read_form(request.form)
    if len(f["prompt"]) < 3:
        return render_template("index.html", settings=SETTINGS, tones=TONES, styles=STYLES, form=f,
                               error="Please enter a story prompt (at least 3 characters)."), 400
    try:
        comic = make_comic(f, panels)
    except RuntimeError as e:
        return render_template("index.html", settings=SETTINGS, tones=TONES, styles=STYLES, form=f, error=str(e)), 502
    return render_template("preview.html", comic=comic)


@app.post("/generate-comic/json")
def generate_json():
    f, panels = read_form(request.get_json(silent=True) or {})
    if len(f["prompt"]) < 3:
        return jsonify(error="prompt is required (min 3 characters)"), 400
    try:
        c = make_comic(f, panels)
    except RuntimeError as e:
        return jsonify(error=str(e)), 502
    return jsonify(id=c["id"], title=c["title"], demo=c["demo"], pdf_url=f"/comic/{c['id']}/pdf",
                   panels=[{k: p[k] for k in ("number", "title", "scene", "caption", "narration", "image_prompt", "image_source")}
                           for p in c["panels"]])


@app.get("/comic/<cid>/pdf")
def comic_pdf(cid):
    c = COMICS.get(cid)
    if not c:
        return render_template("success.html", missing=True), 404
    return send_file(build_pdf(c), mimetype="application/pdf", as_attachment=True, download_name="comic.pdf")


@app.get("/export-success")
def export_success():
    return render_template("success.html", missing=False)


@app.get("/test-image")
def test_image():
    data, src = fetch_image("a red fox standing at the edge of a dark forest, realistic painting", 0, "Test image")
    return send_file(io.BytesIO(data), mimetype="image/jpeg", download_name="test.jpg", max_age=0,
                     etag=False, conditional=False) if request.args.get("raw") else jsonify(
        backend=env("IMAGE_BACKEND", "pollinations"), source_used=src, bytes=len(data), view_raw="/test-image?raw=1")


@app.get("/docs")
def docs():
    return render_template("docs.html")


@app.get("/api/health")
def health():
    return jsonify(ok=True, has_key=bool(env("GEMINI_API_KEY")) and not env("GEMINI_API_KEY").endswith("your_key_here"),
                   env_file_found=os.path.exists(os.path.join(BASE_DIR, ".env")),
                   flash_model=env("GEMINI_FLASH_MODEL", FALLBACK_MODEL), image_backend=env("IMAGE_BACKEND", "pollinations"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(env("PORT", "8000")), debug=False)
