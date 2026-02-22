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
        energy_threshold=0.005,
        on_transcription=None,
        on_speech_start=None,
        on_speech_end=None,
    ):
        self._transcriber = transcriber
        self._silence_frames_needed = silence_timeout_ms // FRAME_MS
        self._min_speech_frames = min_utterance_ms // FRAME_MS
        self._pre_buffer_frames = pre_buffer_ms // FRAME_MS
        self._vad_threshold = vad_threshold
        self._energy_threshold = energy_threshold
        self._on_transcription = on_transcription
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end

        # Queues and state
        self._result_q = queue.Queue()
        self._running = False
        self._stop_event = threading.Event()
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
        self._stop_event.clear()

        self._transcribe_q = queue.Queue()

        # VAD state – reset each start
        self._speech_buffer = []
        self._pre_buffer = collections.deque(maxlen=self._pre_buffer_frames)
        self._silent_frames = 0
        self._is_speaking = False

        t = threading.Thread(target=self._transcribe_worker, daemon=True)
        t.start()
        self._threads.append(t)

        t2 = threading.Thread(target=self._stream_holder, daemon=True)
        t2.start()
        self._threads.append(t2)

    def stop(self):
        """Stop capture and drain the transcription queue."""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
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
        """Called by sounddevice for each audio frame — runs VAD inline."""
        if status:
            print(f"[audio: {status}]", file=sys.stderr)

        frame = indata[:, 0].copy()

        # Cheap energy gate: skip VAD inference on near-silent frames
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < self._energy_threshold:
            if self._is_speaking:
                self._silent_frames += 1
                self._speech_buffer.append(frame)
                if self._silent_frames >= self._silence_frames_needed:
                    self._is_speaking = False
                    self._silent_frames = 0
                    self._vad_model.reset_states()
                    if self._on_speech_end:
                        self._on_speech_end()
                    if len(self._speech_buffer) >= self._min_speech_frames:
                        self._transcribe_q.put(list(self._speech_buffer))
                    self._speech_buffer.clear()
            else:
                self._pre_buffer.append(frame)
            return

        frame_tensor = torch.from_numpy(frame)
        speech_prob = self._vad_model(frame_tensor, SAMPLE_RATE).item()

        if speech_prob >= self._vad_threshold:
            if not self._is_speaking:
                self._is_speaking = True
                if self._on_speech_start:
                    self._on_speech_start()
                self._speech_buffer.extend(self._pre_buffer)
                self._pre_buffer.clear()
            self._speech_buffer.append(frame)
            self._silent_frames = 0

        elif self._is_speaking:
            self._speech_buffer.append(frame)
            self._silent_frames += 1

            if self._silent_frames >= self._silence_frames_needed:
                self._is_speaking = False
                self._silent_frames = 0
                self._vad_model.reset_states()
                if self._on_speech_end:
                    self._on_speech_end()

                if len(self._speech_buffer) >= self._min_speech_frames:
                    self._transcribe_q.put(list(self._speech_buffer))

                self._speech_buffer.clear()

        else:
            self._pre_buffer.append(frame)

    def _stream_holder(self):
        """Open the audio stream and keep it alive until stop is requested."""
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._audio_callback,
        )
        with stream:
            self._stop_event.wait()

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
