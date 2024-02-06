from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import subprocess
from fastapi.middleware.cors import CORSMiddleware
import logging
import shutil
from datetime import datetime
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware

# Add more permissive CORS settings during development
origins = ["*"]

# Add the middleware
app = FastAPI(
    title="DrQA backend API",
    docs_url="/docs",
    middleware=[
        Middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]),
        Middleware(StarletteCORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]),
    ]
)

logging.basicConfig(level=logging.INFO)

def upload_to_s3(file_path, s3_bucket, s3_key):
    command = f"aws s3 cp {file_path} s3://{s3_bucket}/{s3_key}"
    with subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        process.communicate()

@app.post("/upload")
async def upload_file_to_s3(req: Request):
    try:
        # Receive 'id' from the JSON request body
        id = (await req.json())["id"]  # Use await to get the result of the coroutine

        if not id:
            raise HTTPException(status_code=400, detail="Missing 'id' in the request body")

        # Use an absolute path for the file
        current_date = datetime.now().strftime("%m-%d-%Y")
        file_name = os.path.abspath(f"VATest-Evva_Health_{current_date}_{id}.docx")
        s3_key = f"01-028/VATest-Evva_Health_{current_date}_{id}.docx"

        # Copy the existing Word document to the new file name
        original_file_name = "VATest-Evva_Health-2Feb2024.docx"
        shutil.copy(original_file_name, file_name)

        # Upload the file to S3
        upload_to_s3(file_name, "naii-01-dir", s3_key)

        return JSONResponse(content={"message": "File uploaded successfully to S3!"}, status_code=200)
    except HTTPException as http_exc:
        return JSONResponse(content={"message": str(http_exc.detail)}, status_code=http_exc.status_code)
    except Exception:
        return JSONResponse(content={"message": "Internal Server Error"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)
