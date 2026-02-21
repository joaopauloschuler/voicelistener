#!/usr/bin/env python3
"""Terminal voice recognition with Silero VAD + Whisper tiny."""

import sys
import queue
import collections
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

# ── Config ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_MS = 32  # VAD frame size in ms (Silero needs >=512 samples at 16kHz)
FRAME_SAMPLES = 512
SILENCE_TIMEOUT_MS = 600  # silence duration to finalize utterance
MIN_UTTERANCE_MS = 250  # discard very short detections
PRE_BUFFER_MS = 150  # audio to keep before VAD triggers
VAD_THRESHOLD = 0.5

silence_frames_needed = SILENCE_TIMEOUT_MS // FRAME_MS
min_speech_frames = MIN_UTTERANCE_MS // FRAME_MS
pre_buffer_frames = PRE_BUFFER_MS // FRAME_MS


def load_models():
    """Load Silero VAD and Whisper tiny."""
    print("Loading Silero VAD...", flush=True)
    vad_model, vad_utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )

    print("Loading Whisper tiny...", flush=True)
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

    print("Models ready. Listening...\n", flush=True)
    return vad_model, whisper


def transcribe(whisper, audio_frames):
    """Run Whisper on accumulated audio frames."""
    audio = np.concatenate(audio_frames)
    # faster-whisper expects float32 numpy array
    segments, _ = whisper.transcribe(audio, language="en", beam_size=3)
    text = " ".join(seg.text for seg in segments).strip()
    return text


def main():
    vad_model, whisper = load_models()

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio: {status}]", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
    )

    speech_buffer = []
    pre_buffer = collections.deque(maxlen=pre_buffer_frames)
    silent_frames = 0
    is_speaking = False

    try:
        with stream:
            while True:
                frame = audio_q.get()
                frame_tensor = torch.from_numpy(frame)

                speech_prob = vad_model(frame_tensor, SAMPLE_RATE).item()

                if speech_prob >= VAD_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        # Prepend pre-buffer so the start of speech isn't clipped
                        speech_buffer.extend(pre_buffer)
                        pre_buffer.clear()
                        print("* ", end="", flush=True)
                    speech_buffer.append(frame)
                    silent_frames = 0

                elif is_speaking:
                    # Still accumulate during short silence gaps within speech
                    speech_buffer.append(frame)
                    silent_frames += 1

                    if silent_frames >= silence_frames_needed:
                        # Utterance complete
                        is_speaking = False
                        silent_frames = 0

                        if len(speech_buffer) >= min_speech_frames:
                            text = transcribe(whisper, speech_buffer)
                            if text:
                                print(text, flush=True)
                            else:
                                print("(inaudible)", flush=True)
                        else:
                            print("(skip)", flush=True)

                        speech_buffer.clear()

                else:
                    # Not speaking — keep a rolling pre-buffer
                    pre_buffer.append(frame)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
