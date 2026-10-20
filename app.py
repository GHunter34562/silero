from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import io
import soundfile as sf

app = FastAPI()

# Ограничиваем потоки процессора
torch.set_num_threads(2)

# Загружаем более легкую модель v4
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v4_ru',
    trust_repo=True
)
model.to(torch.device('cpu'))

class TTSRequest(BaseModel):
    text: str
    speaker: str = "kseniya"

@app.get("/")
def ping():
    return {"status": "ok"}

@app.post("/generate")
def generate(data: TTSRequest):
    audio = model.apply_tts(
        text=data.text,
        speaker=data.speaker,
        sample_rate=48000
    )
    
    buffer = io.BytesIO()
    sf.write(buffer, audio.numpy(), 48000, format='WAV')
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")
