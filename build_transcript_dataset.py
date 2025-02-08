import logging
from pathlib import Path
from youtubetools.config import ROOT_DIR
import progressbar
import pandas as pd

if __name__ == "__main__":
    collection = "recs_dQw4w9WgXcQ_2_20250204_184511_046575"
    logging.info(f"Starting building dataset for collection: {collection}")

    transcript_dir = Path(ROOT_DIR) / "collections" / collection / "transcripts"
    transcript_files = list(transcript_dir.glob("*.txt"))
    video_ids = [transcript_file.stem.removesuffix("_whisper3") for transcript_file in transcript_files]
    total_videos = len(transcript_files)
    logging.info(f"Found {total_videos} transcripts to process")

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

    dataset = []
    for video_id, transcript_file in zip(video_ids, transcript_files):
        with open(transcript_file, "r") as f:
            transcript = f.read()
            dataset.append(
                {
                    "video_id": video_id,
                    "transcript": transcript,
                })

    df = pd.DataFrame(dataset)
    df.to_csv(Path(ROOT_DIR) / "datasets" / f"{collection}_transcripts.csv", index=False)
