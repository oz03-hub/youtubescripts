import csv
import os
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import progressbar
from queue import Queue
from threading import Thread, Lock
from youtubetools.config import ROOT_DIR
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("whisper_transcription.log"),
        logging.StreamHandler(),
    ],
)

# Global configurations
device = "cuda" if torch.cuda.is_available() else "cpu"
max_threads = min(torch.cuda.device_count() if device == "cuda" else 10, os.cpu_count())
torch_dtype = torch.float16 if device == "cuda" else torch.float32
model_id = "openai/whisper-large-v3"

# Global model and processor to avoid reloading for each thread
model_lock = Lock()
model = None
processor = None


def initialize_model():
    """Initialize the model and processor once"""
    global model, processor
    with model_lock:
        if model is None:
            logging.info(f"Initializing Whisper model on {device} with {torch_dtype}")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                device_map="auto" if device == "cuda" else None,
            )
            processor = AutoProcessor.from_pretrained(model_id)
            logging.info("Model initialization complete")


def transcribe(collection, video_id):
    """Transcribe a single audio file with error handling"""
    try:
        wav_path = (
            Path(ROOT_DIR) / "collections" / collection / "wavs" / f"{video_id}.wav"
        )
        output_path = (
            Path(ROOT_DIR)
            / "collections"
            / collection
            / "transcripts"
            / f"{video_id}_whisper3.txt"
        )

        if not wav_path.exists():
            logging.error(f"WAV file not found: {wav_path}")
            return False

        logging.debug(f"Processing {video_id}")

        with model_lock:
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=device,
            )

            result = pipe(
                str(wav_path),
                generate_kwargs={
                    "max_new_tokens": 400,
                    "return_timestamps": True,
                },
            )

        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        logging.debug(f"Successfully transcribed {video_id}")
        return True

    except Exception as e:
        logging.error(f"Error processing {video_id}: {str(e)}", exc_info=True)
        return False


def worker(q, progress_lock):
    """Worker function with proper error handling and progress updates"""
    while not q.empty():
        try:
            video_id = q.get()
            success = transcribe(collection, video_id)

            with progress_lock:
                progress = (total_videos - q.qsize()) / total_videos * 100
                pbar.update(progress)

            q.task_done()

        except Exception as e:
            logging.error(f"Worker thread error: {str(e)}", exc_info=True)
            q.task_done()


if __name__ == "__main__":
    try:
        collection = "recs_dQw4w9WgXcQ_2_20250204_184511_046575"
        logging.info(f"Starting transcription for collection: {collection}")

        # Initialize model once
        initialize_model()

        # Get wav files
        wav_dir = Path(ROOT_DIR) / "collections" / collection / "wavs"
        wav_files = list(wav_dir.glob("*.wav"))
        video_ids = [wav_file.stem for wav_file in wav_files]

        total_videos = len(wav_files)
        logging.info(f"Found {total_videos} files to process")

        # Initialize progress bar
        pbar = progressbar.ProgressBar(
            maxval=100,
            widgets=[
                "Progress: ",
                progressbar.Percentage(),
                " ",
                progressbar.Bar(),
                " ",
                progressbar.ETA(),
            ],
        ).start()

        # Set up queue and threads
        q = Queue()
        for video_id in video_ids:
            q.put(video_id)

        progress_lock = Lock()
        threads = []
        for i in range(max_threads):
            work_thread = Thread(target=worker, args=(q, progress_lock))
            work_thread.daemon = True
            threads.append(work_thread)
            work_thread.start()
            logging.info(f"Started worker thread {i+1}")

        # Wait for completion
        q.join()
        pbar.finish()
        logging.info("Transcription complete")

    except Exception as e:
        logging.error(f"Main thread error: {str(e)}", exc_info=True)
        raise
