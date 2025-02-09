"""
Downloads .wav files for each video ID in a given CSV file
"""
import argparse
import os
import pandas as pd
import subprocess
import time
from tqdm import tqdm

def download_audio(folder, video_id, two_minutes=True):
    tries = 0
    while tries < 5:
        try:
            query = ['yt-dlp', '-q', '--no-progress', '-o', os.path.join(folder, "wavs",
                                                                         "%(id)s.%(ext)s"), '-x', '--audio-format',
                     'wav', '--audio-quality', '256K', '--cookies-from-browser', 'chrome']
            if two_minutes:
                query += ['--download-sections', '*00:00:00-00:02:00']
            query += ['--ppa', 'ffmpeg:-ar 16000 -ac 1', f'https://www.youtube.com/watch?v={video_id}']
            
            # Run the command and capture output
            result = subprocess.run(query, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Error downloading audio for {video_id}: {result.stderr.decode()}")
            else:
                return os.path.join(folder, "wavs", f"{video_id}.wav")
        except Exception as e:
            print(f"Exception downloading audio for {video_id}: {e}")
            tries += 1
            time.sleep(5)
    return None

parser = argparse.ArgumentParser()
parser.add_argument("--csvfile", type=str, help="Path to the CSV file containing video IDs")
parser.add_argument("--two_minutes", type=bool, default=True, help="Download 2-minute audio")
parser.add_argument("--limit", type=int, default=None, help="Limit the number of videos to download")
args = parser.parse_args()

# Create a directory with the same name as the CSV file (without extension)
csv_directory = os.path.splitext(args.csvfile)[0]
os.makedirs(csv_directory, exist_ok=True)

# Read video IDs from the CSV file
video_ids = pd.read_csv(args.csvfile).iloc[:, 0].tolist()  # Assuming the column is named 'video_id'

# Download audio for each video ID
for video_id in tqdm(video_ids[:args.limit]):
    download_audio(csv_directory, video_id, args.two_minutes)
