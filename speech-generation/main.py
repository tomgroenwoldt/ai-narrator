import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from TTS.api import TTS

app = FastAPI()

# Load TTS with model as global state
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

audio_file_name = "output.wav"

@app.get("/process-sentence/")
def process_string(sentence: str):
    # Convert text to audio
    tts.tts_to_file(text=sentence, speaker_wav="sample.wav", language="en", file_path=audio_file_name)

    # Send the WAV file as a response
    return FileResponse(audio_file_name, media_type="audio/wav")
