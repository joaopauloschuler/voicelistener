"""VoiceListener: audio capture + VAD + transcription threading."""

import sys
import queue
import collections
import threading

import numpy as np
import sounddevice as sd
import torch


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_MS = 32


class VoiceListener:
    """Streams microphone audio through Silero VAD and yields transcriptions.

    Usage::

        for text in VoiceListener(transcriber=my_transcriber):
            print(text)
    """

    def __init__(
        self,
        transcriber,
        *,
        silence_timeout_ms=2000,
        min_utterance_ms=250,
        pre_buffer_ms=150,
        vad_threshold=0.5,
        on_transcription=None,
    ):
        self._transcriber = transcriber
        self._silence_frames_needed = silence_timeout_ms // FRAME_MS
        self._min_speech_frames = min_utterance_ms // FRAME_MS
        self._pre_buffer_frames = pre_buffer_ms // FRAME_MS
        self._vad_threshold = vad_threshold
        self._on_transcription = on_transcription

        # Queues and state
        self._audio_q = queue.Queue()
        self._result_q = queue.Queue()
        self._running = False
        self._threads = []

        # Load Silero VAD
        self._vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )

    # ── Public API ────────────────────────────────────────────────────────

    def start(self):
        """Start audio capture and transcription threads."""
        if self._running:
            return
        self._running = True

        self._transcribe_q = queue.Queue()

        t = threading.Thread(target=self._transcribe_worker, daemon=True)
        t.start()
        self._threads.append(t)

        t2 = threading.Thread(target=self._vad_loop, daemon=True)
        t2.start()
        self._threads.append(t2)

    def stop(self):
        """Stop capture and drain the transcription queue."""
        if not self._running:
            return
        self._running = False
        try:
            self._transcribe_q.put(None)  # sentinel for worker
        except Exception:
            pass
        for t in self._threads:
            try:
                t.join(timeout=2)
            except (KeyboardInterrupt, Exception):
                break
        self._threads.clear()

    def __iter__(self):
        self.start()
        try:
            while self._running or not self._result_q.empty():
                try:
                    text = self._result_q.get(timeout=0.2)
                    yield text
                except queue.Empty:
                    continue
        except GeneratorExit:
            self.stop()

    # ── Internal ──────────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio: {status}]", file=sys.stderr)
        self._audio_q.put(indata[:, 0].copy())

    def _vad_loop(self):
        """Capture audio, run VAD, and dispatch complete utterances."""
        speech_buffer = []
        pre_buffer = collections.deque(maxlen=self._pre_buffer_frames)
        silent_frames = 0
        is_speaking = False

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._audio_callback,
        )

        with stream:
            while self._running:
                try:
                    frame = self._audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                frame_tensor = torch.from_numpy(frame)
                speech_prob = self._vad_model(frame_tensor, SAMPLE_RATE).item()

                if speech_prob >= self._vad_threshold:
                    if not is_speaking:
                        is_speaking = True
                        speech_buffer.extend(pre_buffer)
                        pre_buffer.clear()
                    speech_buffer.append(frame)
                    silent_frames = 0

                elif is_speaking:
                    speech_buffer.append(frame)
                    silent_frames += 1

                    if silent_frames >= self._silence_frames_needed:
                        is_speaking = False
                        silent_frames = 0

                        if len(speech_buffer) >= self._min_speech_frames:
                            self._transcribe_q.put(list(speech_buffer))

                        speech_buffer.clear()

                else:
                    pre_buffer.append(frame)

    def _transcribe_worker(self):
        """Pull audio chunks from the queue and transcribe them."""
        while True:
            audio_frames = self._transcribe_q.get()
            if audio_frames is None:
                break
            audio = np.concatenate(audio_frames)
            text = self._transcriber.transcribe(audio)
            if text:
                self._result_q.put(text)
                if self._on_transcription:
                    self._on_transcription(text)
