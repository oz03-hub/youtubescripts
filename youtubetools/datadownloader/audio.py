import os
import time
import subprocess
from youtubetools.config import ROOT_DIR
from youtubetools.logger import log_error


def download_audio_track(collection: str, video_id: str, options: dict = None) -> str:
    if os.path.isfile(os.path.join(ROOT_DIR, "collections", collection, f"{video_id}.wav")):
        return os.path.join(ROOT_DIR, "collections", collection, f"{video_id}.wav")

    tries = 0
    while tries < 5:
        try:
            subprocess.run(
                ['yt-dlp', '-q', '-o', os.path.join(ROOT_DIR, "collections", collection, "wavs", "%(id)s.%(ext)s"),
                 '-x', '--audio-format', 'wav', '--audio-quality', '256K', '--ppa', 'ffmpeg:-ar 16000 -ac 1',
                 f'https://www.youtube.com/watch?v={video_id}'])
            return os.path.join(ROOT_DIR, "collections", collection, f"{video_id}.wav")
        except Exception as e:
            log_error(collection, video_id, "datadownloader_audio", e)
            tries += 1
            time.sleep(5)
    return ""
