# whisper_api.py
import io
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from vllm import AsyncLLMEngine, AsyncEngineArgs
from vllm.inputs.data import SpeechToTextInput

# ── engine config ─────────────────────────────────────────────────────────────
ENGINE_ARGS = AsyncEngineArgs(
    model="openai/whisper-large-v3-turbo",
    task="transcription",
    max_model_len=448,
)
engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

app = FastAPI()

# ── audio helpers ─────────────────────────────────────────────────────────────
PAD_SECONDS    = 0.5
INITIAL_PROMPT = "Transcribe all spoken words exactly as heard."


def pad_audio(audio: np.ndarray, sr: int, pad_seconds: float = PAD_SECONDS) -> np.ndarray:
    """Prepend silence — prevents Whisper clipping the first words."""
    silence = np.zeros(int(sr * pad_seconds), dtype=audio.dtype)
    return np.concatenate([silence, audio])


def to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode numpy array → WAV bytes (no temp files)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


# ── endpoint ──────────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        raw_bytes = await file.read()
        buf = io.BytesIO(raw_bytes)
        audio, sr = sf.read(buf, dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read audio: {e}")

    # Fix 1: pad to prevent missing initial words
    audio = pad_audio(audio, sr, PAD_SECONDS)
    wav_bytes = to_wav_bytes(audio, sr)

    # Fix 2: initial prompt (decoder warm-start) + suppress_tokens (no profanity masking)
    vllm_input = SpeechToTextInput(
        audio=wav_bytes,
        prompt=INITIAL_PROMPT,           # decoder warm-start
    )

    sampling_params = SamplingParams(
        suppress_tokens=[],              # disable profanity token suppression
        max_tokens=448,
    )

    # stream and collect
    transcript = ""
    async for output in engine.generate(vllm_input, sampling_params, request_id="req-1"):
        if output.finished:
            transcript = output.outputs[0].text
            break

    return {"transcript": transcript}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
