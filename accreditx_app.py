from __future__ import annotations

import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

from model_3d import Brain3DCNN
from train import BrainCNN

try:
    from authlib.integrations.flask_client import OAuth
except ImportError:
    OAuth = None

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "accreditx.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_2D = ROOT / "artifacts_80min" / "best_model.pt"
MODEL_3D = ROOT / "artifacts_3d" / "best_model.pt"
LIVER_MODEL = ROOT / "models" / "liver_3d" / "Leatest1.pth"
app = Flask(__name__, template_folder="accreditx_templates", static_folder="accreditx_static")
app.secret_key = os.getenv("ACCREDITX_SECRET_KEY", "change-this-local-demo-secret")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
MODELS = {}
oauth = OAuth(app) if OAuth else None
ALLOWED_SCAN_EXTENSIONS = (".png", ".jpg", ".jpeg", ".nii", ".nii.gz")


def google_configured():
    return bool(os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"))


if oauth and google_configured():
    oauth.register(name="google", client_id=os.getenv("GOOGLE_CLIENT_ID"), client_secret=os.getenv("GOOGLE_CLIENT_SECRET"), server_metadata_url="https://accounts.google.com/.well-known/openid-configuration", client_kwargs={"scope": "openid email profile"})


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
          id INTEGER PRIMARY KEY AUTOINCREMENT, patient_code TEXT UNIQUE NOT NULL,
          display_name TEXT NOT NULL, dob TEXT, sex TEXT, mrn_last4 TEXT,
          contact_note TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scans (
          id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL,
          cancer_type TEXT NOT NULL, filename TEXT NOT NULL, modality TEXT NOT NULL,
          result TEXT NOT NULL, confidence REAL, priority TEXT NOT NULL,
          notes TEXT, device TEXT NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
        CREATE TABLE IF NOT EXISTS audit_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT NOT NULL,
          action TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL
        );
        """)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit(action: str, detail: dict):
    with db() as conn:
        conn.execute("INSERT INTO audit_log(actor, action, detail, created_at) VALUES (?, ?, ?, ?)", (session.get("user", "system"), action, json.dumps(detail), now()))


def logged_in():
    return "user" in session


def load_2d():
    if "2d" in MODELS: return MODELS["2d"]
    if not MODEL_2D.exists(): return None, None
    ckpt = torch.load(MODEL_2D, map_location=DEVICE, weights_only=True)
    model = BrainCNN(len(ckpt["classes"])).to(DEVICE); model.load_state_dict(ckpt["model"]); model.eval(); MODELS["2d"] = (model, ckpt); return model, ckpt


def load_3d():
    if "3d" in MODELS: return MODELS["3d"]
    if not MODEL_3D.exists(): return None, None
    ckpt = torch.load(MODEL_3D, map_location=DEVICE, weights_only=True)
    model = Brain3DCNN(len(ckpt.get("classes", ["no_tumor", "tumor"]))).to(DEVICE); model.load_state_dict(ckpt["model"]); model.eval(); MODELS["3d"] = (model, ckpt); return model, ckpt


def normalize_volume(volume):
    volume = np.nan_to_num(volume.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); lo, hi = np.percentile(volume, (1, 99))
    if hi <= lo: lo, hi = float(volume.min()), float(volume.max() + 1.0)
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0)


def run_scan(cancer_type: str, uploaded):
    filename = uploaded.filename or "scan"
    if cancer_type == "brain_2d":
        model, ckpt = load_2d()
        if model is None: return {"result": "2D model unavailable", "confidence": None, "modality": "2D MRI", "device": str(DEVICE)}
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((ckpt["image_size"], ckpt["image_size"])), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        with torch.inference_mode(): probs = model(tfm(image).unsqueeze(0).to(DEVICE)).softmax(1)[0]
        idx = int(probs.argmax()); return {"result": ckpt["classes"][idx], "confidence": round(float(probs[idx]) * 100, 2), "modality": "2D MRI", "device": str(DEVICE)}
    if cancer_type == "brain_3d":
        return {"result": "3D model pending validation", "confidence": None, "modality": "3D brain MRI", "device": str(DEVICE)}
    if cancer_type == "liver":
        if not LIVER_MODEL.exists(): return {"result": "Liver model unavailable", "confidence": None, "modality": "3D liver CT", "device": str(DEVICE)}
        return {"result": "Liver CT received for model review", "confidence": None, "modality": "3D liver CT", "device": str(DEVICE)}
    raise ValueError("Unknown cancer model")


@app.context_processor
def context(): return {"device": str(DEVICE), "app_name": "Accredix"}


@app.get("/")
def home(): return redirect(url_for("dashboard" if logged_in() else "login"))


@app.get("/login")
def login(): return render_template("login.html", google_ready=bool(oauth and google_configured()))


@app.post("/demo-login")
def demo_login():
    session["user"] = "demo@accreditx.local"; session["role"] = "research-admin"; audit("demo_login", {}); return redirect(url_for("dashboard"))

@app.get("/auth/google")
def google_login():
    if not oauth or not google_configured():
        flash("Google OAuth is not configured. Use the local demo login or add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.", "error")
        return redirect(url_for("login"))
    return oauth.google.authorize_redirect(url_for("google_callback", _external=True))

@app.get("/auth/callback")
def google_callback():
    if not oauth or not google_configured():
        flash("Google OAuth is not configured. Use the local demo login.", "error")
        return redirect(url_for("login"))
    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo", {})
    except Exception:
        flash("Google sign-in could not be completed. Check the OAuth redirect URI and try again.", "error")
        return redirect(url_for("login"))
    session["user"] = userinfo.get("email", "google-user"); session["role"] = "research-user"; audit("google_login", {"provider": "google"}); return redirect(url_for("dashboard"))


@app.get("/logout")
def logout(): session.clear(); return redirect(url_for("login"))


@app.get("/dashboard")
def dashboard():
    if not logged_in(): return redirect(url_for("login"))
    with db() as conn:
        patients = conn.execute("SELECT p.*, COUNT(s.id) scan_count FROM patients p LEFT JOIN scans s ON s.patient_id=p.id GROUP BY p.id ORDER BY p.created_at DESC").fetchall()
        priority = conn.execute("SELECT s.*, p.patient_code, p.display_name FROM scans s JOIN patients p ON p.id=s.patient_id WHERE s.priority='high' ORDER BY s.created_at DESC").fetchall()
        recent = conn.execute("SELECT s.*, p.patient_code, p.display_name FROM scans s JOIN patients p ON p.id=s.patient_id ORDER BY s.created_at DESC LIMIT 12").fetchall()
    return render_template("dashboard.html", patients=patients, priority=priority, recent=recent, models={"brain_2d": MODEL_2D.exists(), "brain_3d": MODEL_3D.exists(), "liver": LIVER_MODEL.exists()})


@app.post("/patients")
def add_patient():
    if not logged_in(): return redirect(url_for("login"))
    display_name = request.form.get("display_name", "").strip(); code = request.form.get("patient_code", "").strip() or "AX-" + uuid.uuid4().hex[:8].upper()
    if not display_name: flash("Patient display name is required.", "error"); return redirect(url_for("dashboard"))
    if len(code) > 64:
        flash("Patient code must be 64 characters or fewer.", "error")
        return redirect(url_for("dashboard"))
    with db() as conn:
        try:
            conn.execute("INSERT INTO patients(patient_code, display_name, dob, sex, mrn_last4, contact_note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)", (code, display_name, request.form.get("dob", ""), request.form.get("sex", ""), request.form.get("mrn_last4", ""), request.form.get("contact_note", ""), now()))
        except sqlite3.IntegrityError:
            flash("That patient code already exists. Choose a different code.", "error")
            return redirect(url_for("dashboard"))
    audit("patient_created", {"patient_code": code}); flash("Patient added to the local registry.", "success"); return redirect(url_for("dashboard"))


@app.post("/scans")
def add_scan():
    if not logged_in(): return redirect(url_for("login"))
    uploaded = request.files.get("scan"); cancer_type = request.form.get("cancer_type", "brain_2d"); patient_id = request.form.get("patient_id")
    if not uploaded or not patient_id:
        flash("Choose a patient and scan.", "error")
        return redirect(url_for("dashboard"))
    filename = secure_filename(uploaded.filename or "")
    if not filename or not filename.lower().endswith(ALLOWED_SCAN_EXTENSIONS):
        flash("Unsupported scan file. Use PNG, JPG, NIfTI, or NIfTI.GZ.", "error")
        return redirect(url_for("dashboard"))
    try:
        with db() as conn:
            if conn.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone() is None:
                flash("Choose a valid patient from the registry.", "error")
                return redirect(url_for("dashboard"))
        result = run_scan(cancer_type, uploaded)
    except Exception as exc: flash(f"Scan could not be processed: {exc}", "error"); return redirect(url_for("dashboard"))
    with db() as conn:
        conn.execute("INSERT INTO scans(patient_id, cancer_type, filename, modality, result, confidence, priority, notes, device, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (patient_id, cancer_type, filename, result["modality"], result["result"], result["confidence"], request.form.get("priority", "routine"), request.form.get("notes", ""), result["device"], now()))
    audit("scan_recorded", {"cancer_type": cancer_type, "priority": request.form.get("priority", "routine")}); flash("Scan recorded in the local audit trail.", "success"); return redirect(url_for("dashboard"))


@app.get("/api/health")
def health(): return jsonify({"status": "ok", "app": "Accredix", "modes": ["2d", "3d", "liver"], "device": str(DEVICE), "models": {"brain_2d": MODEL_2D.exists(), "brain_3d": MODEL_3D.exists(), "liver": LIVER_MODEL.exists()}})


init_db()
if __name__ == "__main__": app.run(host="127.0.0.1", port=5000, debug=False)
