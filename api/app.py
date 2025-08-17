import json
import shutil

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, Response, stream_with_context
import torch
from PIL import Image
import os, sys
from pathlib import Path
import torchvision.transforms as transforms
from flask_sqlalchemy import SQLAlchemy
from git import Repo
from sqlalchemy import func
import uuid
import subprocess
from datetime import datetime

# Make sure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from src.model import CatClassifier

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# Load DB
# BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'logs.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
# ====== DB Model ======
class RequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(300), nullable=True)
    predicted_label = db.Column(db.String(80), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    folder_name = db.Column(db.String(80), nullable=True)
    image_name = db.Column(db.String(300), nullable=True)
    true_label = db.Column(db.String(80), nullable=True)

with app.app_context():
    db.create_all()

# Load model
model = CatClassifier()
model.load_state_dict(torch.load("../model.pth", map_location="cpu"))
model.eval()

# Routes
@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    image = Image.open(file.stream).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
    prediction = torch.argmax(output, 1).item()
    probs = torch.softmax(output, dim=1)
    prediction_idx = torch.argmax(probs, 1).item()
    confidence = probs[0][prediction_idx].item()

    pred_label = "Cat" if prediction_idx == 1 else "Not Cat"

    # Chọn thư mục đích theo prediction
    folder_name = "Cat" if prediction_idx == 1 else "Not_Cat"
    target_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    os.makedirs(target_folder, exist_ok=True)

    # Đặt tên file an toàn
    ext = ".jpg"
    fname = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(target_folder, fname)

    # Lưu ảnh chuẩn JPEG
    image.save(save_path, format="JPEG")

    # Save log
    log = RequestLog(
        username=request.form.get("username", "anonymous"),
        folder_name=folder_name,
        image_name=fname,
        predicted_label=pred_label,
        confidence=confidence,
        true_label=''
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({"prediction": "Cat" if prediction == 1 else "Not Cat"})

@app.route("/uploads/<folder>/<path:filename>")
def uploaded_file(folder, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], folder), filename)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/metrics/accuracy")
def api_accuracy():
    total_logs = RequestLog.query.count()
    if total_logs == 0:
        return jsonify({"average_accuracy": 0, "total_logs": 0})

    avg_conf = db.session.query(db.func.avg(RequestLog.confidence)).scalar()
    return jsonify({
        "average_accuracy": round(avg_conf * 100, 2),
        "total_logs": total_logs
    })

@app.route('/api/metrics/requests_by_day')
def api_requests_by_day():
    rows = db.session.query(
        func.date(RequestLog.timestamp).label('day'),
        func.count(RequestLog.id).label('count')
    ).group_by(func.date(RequestLog.timestamp)).order_by(func.date(RequestLog.timestamp)).all()
    return jsonify([{"day": r.day, "count": int(r.count)} for r in rows])


@app.route('/api/logs/recent')
def api_recent_logs():
    rows = RequestLog.query.order_by(RequestLog.timestamp.desc()).limit(50).all()
    data = []
    for r in rows:
        data.append({
            "id": r.id,
            "username": r.username,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "image_url": url_for('uploaded_file', folder=r.folder_name, filename=r.image_name),
            "predicted_label": r.predicted_label,
            "confidence": r.confidence,
            "true_label": r.true_label # Include true_label
        })
    return jsonify(data)

# ----- SSE stream endpoint: run tasks and stream progress -----
@app.route('/stream/push-retrain')
def stream_push_retrain():
    """
    SSE endpoint. When client opens EventSource to this URL, server will
    execute steps sequentially and yield JSON messages as SSE 'data' events.
    """
    def send_json(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate():
        # Step 1: Move user's image to Train Data
        yield send_json({"step": "move_images", "status": "started", "message": "Starting move images..."})
        try:
            moved = 0
            # Get all logs with true_label
            annotated_logs = RequestLog.query.filter(RequestLog.true_label.isnot(None)).all()
            dst_base = os.path.join(BASE_DIR, "..", "data", "images")
            os.makedirs(dst_base, exist_ok=True)

            for log in annotated_logs:
                true_label_folder = "Cat" if log.true_label == "Cat" else "Not_Cat"
                src_path = os.path.join(app.config['UPLOAD_FOLDER'], log.folder_name, log.image_name)
                dst_folder = os.path.join(dst_base, true_label_folder)
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, log.image_name)

                if os.path.exists(src_path):
                    try:
                        shutil.copy2(src_path, dst_path)
                        moved += 1
                        # log.true_label = None  # Reset annotation after moving
                        log.folder_name = true_label_folder
                        db.session.commit()
                    except Exception as e:
                        yield send_json({"step": "move_images", "status": "warning",
                                         "message": f"Skipped {log.image_name}: {str(e)}"})

            yield send_json({"step": "move_images", "status": "success", "message": f"Moved {moved} files."})
        except Exception as e:
            yield send_json({"step": "move_images", "status": "failed", "message": str(e)})
            yield send_json({"event": "finished", "success": False})
            return

        # Step 2: git add
        yield send_json({"step": "git_add", "status": "started", "message": "Running git add ."})
        try:
            p = subprocess.run(["git", "add", "."], cwd=BASE_DIR, capture_output=True, text=True)
            if p.returncode == 0:
                yield send_json({"step": "git_add", "status": "success", "message": p.stdout.strip() or "git add ok"})
            else:
                # show git stderr but continue (treat as failure)
                yield send_json({"step": "git_add", "status": "failed", "message": p.stderr.strip() or p.stdout.strip()})
                yield send_json({"event": "finished", "success": False})
                return
        except Exception as e:
            yield send_json({"step": "git_add", "status": "failed", "message": str(e)})
            yield send_json({"event": "finished", "success": False})
            return

        # Step 3: git commit
        yield send_json({"step": "git_commit", "status": "started", "message": "Running git commit"})
        try:
            p = subprocess.run(["git", "commit", "-m", "Push training data"], cwd=BASE_DIR, capture_output=True, text=True)
            combined = (p.stdout or "") + (p.stderr or "")
            if p.returncode == 0:
                yield send_json({"step": "git_commit", "status": "success", "message": combined.strip() or "git commit ok"})
            else:
                # sometimes git commit fails when nothing to commit — treat that as success with message
                if "nothing to commit" in combined.lower():
                    yield send_json({"step": "git_commit", "status": "success", "message": "Nothing to commit"})
                else:
                    yield send_json({"step": "git_commit", "status": "failed", "message": combined.strip()})
                    yield send_json({"event": "finished", "success": False})
                    return
        except Exception as e:
            yield send_json({"step": "git_commit", "status": "failed", "message": str(e)})
            yield send_json({"event": "finished", "success": False})
            return

        # Step 4: git push
        yield send_json({"step": "git_push", "status": "started", "message": "Running git push"})
        try:
            repo_path = os.path.dirname(BASE_DIR)

            # Push branch main
            result = subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                yield send_json({
                    "step": "git_push",
                    "status": "success",
                    "message": result.stdout.strip() or "git push ok"
                })
            else:
                yield send_json({
                    "step": "git_push",
                    "status": "failed",
                    "message": result.stderr.strip() or "git push failed"
                })
                yield send_json({"event": "finished", "success": False})
                return

        except Exception as e:
            yield send_json({"step": "git_push", "status": "failed", "message": str(e)})
            yield send_json({"event": "finished", "success": False})
            return

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"  # turn off buffering for some proxies
    }
    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers=headers)

@app.route('/api/logs/<int:log_id>/annotate', methods=['POST'])
def annotate(log_id):
    true_label = request.json.get('true_label')
    log = RequestLog.query.get_or_404(log_id)
    log.true_label = true_label
    db.session.commit()
    return jsonify({"status": "ok"})

@app.route("/api/recent-uploads")
def api_recent_uploads():
    limit = int(request.args.get("limit", 10))  # Mặc định lấy 10 bản ghi
    logs = RequestLog.query.order_by(RequestLog.timestamp.desc()).limit(limit).all()

    if not logs:
        return jsonify({"recent_uploads": []})

    uploads = []
    for log in logs:
        uploads.append({
            "filename": log.filename,
            "prediction": log.predicted_label,
            "confidence": round(log.confidence, 4) if log.confidence is not None else None,
            "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S") if log.timestamp else None
        })

    return jsonify({"recent_uploads": uploads})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
