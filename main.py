from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import subprocess
import logging
import shutil
from datetime import datetime
from docx import Document

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logging.basicConfig(level=logging.INFO)

def upload_to_s3(file_path, s3_bucket, s3_key):
    command = f"aws s3 cp {file_path} s3://{s3_bucket}/{s3_key}"
    subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

@app.route("/")
def index():
    return "S3 API Online"

@app.route("/uploads3", methods=["POST"])
def upload_file_to_s3():
    try:
        # Receive 'id' from the JSON request body
        id = request.json.get("id")

        if not id:
            raise ValueError("Missing 'id' in the request body")

        # Use an absolute path for the file
        current_date = datetime.now().strftime("%m-%d-%Y")
        file_name = os.path.abspath(f"VATest-Evva_Health_{current_date}_{id}.docx")
        s3_key = f"01-028/VATest-Evva_Health_{current_date}_{id}.docx"

        # Create a new Word document with content "evva health"
        document = Document()
        document.add_paragraph("evva health")
        document.save(file_name)

        # Upload the file to S3
        upload_to_s3(file_name, "naii-01-dir", s3_key)

        return jsonify({"message": "File uploaded successfully to S3!"}), 200
    except ValueError as value_error:
        return jsonify({"message": str(value_error)}), 400
    except Exception as e:
        print("error:", e)
        return jsonify({"message": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
