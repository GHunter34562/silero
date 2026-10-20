from fastapi import FastAPI, Response
from pydantic import BaseModel
import torch
import io
import soundfile as sf

app = FastAPI()

# Ограничиваем CPU, чтобы не вылезти за бесплатные лимиты
torch.set_num_threads(2)

# Загружаем модель Silero TTS v5
model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='ru',
    speaker='v5_4_ru'
)
model.to(torch.device('cpu'))

class TTSRequest(BaseModel):
    text: str
    speaker: str = "kseniya" # Доступны: kseniya, baya, xenia, eugene, aidar

@app.post("/generate")
def generate(data: TTSRequest):
    # Генерация звука
    audio = model.apply_tts(
        text=data.text,
        speaker=data.speaker,
        sample_rate=48000
    )
    
    # Сохраняем в байтовый буфер WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio.numpy(), 48000, format='WAV')
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")