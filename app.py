from pathlib import Path
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from utils.analysis import analyze_media

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "webp", "wav", "mp3", "flac", "mp4", "mov", "avi", "mkv"}
MAX_MB = 100

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/analyze")
def analyze():
    started = time.perf_counter()
    uploaded = request.files.get("media")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Choose an image, audio file, or video first."), 400
    if not allowed_file(uploaded.filename):
        return jsonify(error="Unsupported format. Upload JPG, PNG, WEBP, WAV, MP3, FLAC, MP4, MOV, AVI, or MKV."), 415

    safe_name = secure_filename(uploaded.filename)
    # Avoid collisions while preserving a readable filename.
    destination = UPLOAD_DIR / f"{int(time.time() * 1000)}_{safe_name}"
    uploaded.save(destination)
    try:
        result = analyze_media(destination)
        result["filename"] = safe_name
        result["processing_seconds"] = round(time.perf_counter() - started, 2)
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Media analysis failed")
        return jsonify(error="The file could not be analyzed. Check the format and installed dependencies."), 500
    finally:
        # Remove uploaded file after processing; this keeps the demo privacy-friendly.
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass

@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File is too large. Maximum size is {MAX_MB} MB."), 413

if __name__ == "__main__":
    app.run(debug=True)
