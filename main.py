#!/usr/bin/env python
"""
    main.py:

    Author: Matt Freeland

    Email: matthew_freeland@yahoo.co.uk

    Created: 25/06/2025

    Version: 0.1

    Description:
        hosts a fastapi server that allows users to upload a video, track processing progress,
        and download results (annotated video and log).

    Change History:
        0.1: Created.
"""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from uuid import uuid4
from pathlib import Path
import shutil
import threading

from lab_monitor.pipeline import process_video

# Define project directories
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
LOG_DIR = DATA_DIR / "logs"
PROGRESS = {}

# Ensure directories exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create FastAPI app with custom metadata
app = FastAPI(
    title="Lab Monitor Video Processing API",
    description="Upload a video, track processing progress, and download results (annotated video and log).",
    version="0.1.0"
)


@app.post("/upload/", summary="Upload a video for processing")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
        Upload a video file and begin background processing.
        Returns a `job_id` to track progress and retrieve results.
    """
    job_id = str(uuid4())
    video_path = UPLOAD_DIR / f"{job_id}.mp4"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    log_path = LOG_DIR / f"{job_id}.csv"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    PROGRESS[job_id] = 0

    def run_job():
        process_video(
            str(video_path),
            str(output_path),
            str(log_path),
            lambda p: PROGRESS.update({job_id: p})
        )
        PROGRESS[job_id] = 100  # Mark as complete

    threading.Thread(target=run_job, daemon=True).start()

    return {"job_id": job_id}


@app.get("/status/{job_id}", summary="Check job progress")
def check_status(job_id: str):
    """
        Check the current progress of a video processing job.
    """
    progress = PROGRESS.get(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return {"job_id": job_id, "progress": progress}


@app.get("/download/video/{job_id}", summary="Download the processed video")
def download_video(job_id: str):
    """
        Download the annotated video for a completed job.
    """
    path = OUTPUT_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found.")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/download/log/{job_id}", summary="Download the event log")
def download_log(job_id: str):
    """
        Download the log file (CSV) generated during video processing.
    """
    path = LOG_DIR / f"{job_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Log file not found.")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}.csv")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
