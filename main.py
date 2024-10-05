import pyaudio
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import threading
import queue
import logging
from collections import deque
import time
import re

# Global variable to store incomplete sentences
incomplete_sentence = ""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranscriptionService:
    """Manages the Whisper Turbo model and transcription process."""
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "openai/whisper-large-v3-turbo"

        logger.info(f"Initializing Whisper model on {self.device}")
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        self.running = False  # Controls the transcription loop
        self.max_input_length = 30 * 16000  # 30 seconds at 16kHz

    def start(self, audio_queue, callback):
        """Starts the transcription service in a separate thread."""
        self.audio_queue = audio_queue
        self.callback = callback
        self.running = True
        threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self):
        """Core method for processing audio and generating transcriptions."""
        while self.running:
            try:
                audio_data, force = self.audio_queue.get(timeout=1)

                # Pad or truncate audio data
                if len(audio_data) < self.max_input_length:
                    padding = np.zeros(self.max_input_length - len(audio_data))
                    audio_data = np.concatenate([audio_data, padding])
                else:
                    audio_data = audio_data[:self.max_input_length]

                # Create attention mask
                attention_mask = np.ones_like(audio_data)
                attention_mask[len(audio_data):] = 0

                input_features = self.processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(self.device)

                attention_mask = torch.from_numpy(attention_mask).to(self.device).unsqueeze(0)

                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    language="en",
                    task="transcribe",
                    attention_mask=attention_mask
                )

                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                if transcription.strip():
                    self.callback(transcription.strip(), force)

            except queue.Empty:
                continue

    def stop(self):
        """Stops the transcription service."""
        self.running = False

class AudioRecorder:
    """Handles continuous audio capture and buffering with adaptive speech detection."""
    def __init__(self):
        self.sample_rate = 16000  # Audio sample rate in Hz
        self.chunk_size = 1024  # Number of audio frames per buffer
        self.audio_queue = queue.Queue()  # Queue for audio data to be transcribed
        self.running = False  # Controls the recording loop
        self.buffer = deque(maxlen=None)  # Stores audio data, unlimited size
        self.lock = threading.Lock()  # Thread synchronization for buffer access

        # Speech detection parameters
        self.energy_threshold = 0.0003  # Lowered for more sensitivity
        self.speech_start_threshold = 0.001  # Lowered for quicker detection
        self.speech_end_threshold = 0.0005  # Lowered for quicker end detection
        self.min_pause_length = 0.3  # Shortened to detect brief pauses
        self.max_speech_length = 120  # Increased to allow for longer continuous speech
        self.speech_timeout = 1.0  # Shortened to be more responsive
        self.continuous_speech_threshold = 300  # Increased to allow for very long speech

        # Adaptive parameters
        self.speech_duration_history = deque(maxlen=10)  # Stores recent speech durations
        self.pause_duration_history = deque(maxlen=10)  # Stores recent pause durations
        self.adaptive_factor = 1.5  # Factor for adjusting thresholds based on history
        self.min_max_speech_length = 10  # Minimum allowed value for max_speech_length
        self.max_max_speech_length = 90  # Maximum allowed value for max_speech_length
        self.min_speech_timeout = 0.5  # Minimum allowed value for speech_timeout
        self.max_speech_timeout = 5.0  # Maximum allowed value for speech_timeout

        # Speech tracking
        self.is_speech = False  # Indicates if speech is currently detected
        self.speech_start_time = None  # Timestamp of when current speech segment started
        self.last_speech_end_time = None  # Timestamp of when last speech segment ended

        # Idea segment parameters
        self.idea_timeout = 5.0  # Time to wait before considering an idea complete
        self.current_idea_buffer = []

        self.max_buffer_size = 30 * 16000  # 30 seconds at 16kHz
        self.current_buffer = np.array([], dtype=np.float32)

    def start(self, transcription_callback):
        """Starts audio recording and transcription services."""
        self.running = True
        self.transcription_service = TranscriptionService()
        self.transcription_service.start(self.audio_queue, transcription_callback)
        threading.Thread(target=self._record, daemon=True).start()

    def stop(self):
        """Stops audio recording and transcription services."""
        self.running = False
        self.transcription_service.stop()

    def _record(self):
        """Manages PyAudio stream for continuous audio recording."""
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            stream.start_stream()
            logger.info("Audio stream started")

            while self.running:
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in audio recording: {str(e)}", exc_info=True)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            logger.info("Audio stream stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Processes incoming audio data and manages the buffer."""
        if status:
            logger.warning(f"PyAudio callback status: {status}")

        audio_data = np.frombuffer(in_data, dtype=np.float32)
        energy = np.mean(np.abs(audio_data))
        current_time = time.time()

        self.current_buffer = np.concatenate([self.current_buffer, audio_data])

        if energy > self.speech_start_threshold:
            if not self.is_speech:
                self.is_speech = True
                self.speech_start_time = current_time
            self.last_speech_end_time = current_time
        elif energy < self.speech_end_threshold:
            if self.is_speech and (current_time - self.last_speech_end_time) > self.min_pause_length:
                self.is_speech = False
                self._process_buffer()

        if len(self.current_buffer) >= self.max_buffer_size:
            self._process_buffer(force=True)

        return (in_data, pyaudio.paContinue)

    def _process_buffer(self, force=False):
        if len(self.current_buffer) > 0:
            self.audio_queue.put((self.current_buffer.copy(), force))
            self.current_buffer = np.array([], dtype=np.float32)

        self.speech_start_time = None
        self.last_speech_end_time = None
        self.is_speech = False

    def _process_speech_segment(self, duration, force=False):
        with self.lock:
            if len(self.buffer) > 0:
                segment = np.array(self.buffer)
                self.current_idea_buffer.append(segment)
                self.buffer.clear()

        self._adjust_max_speech_length(duration)

        if force:
            self._process_idea_buffer(force=True)

    def _check_idea_completion(self, current_time):
        if self.last_speech_end_time and (current_time - self.last_speech_end_time) > self.idea_timeout:
            self._process_idea_buffer()

    def _process_idea_buffer(self, force=False):
        if self.current_idea_buffer:
            complete_idea = np.concatenate(self.current_idea_buffer)
            self.audio_queue.put((complete_idea, force))
            self.current_idea_buffer = []

    def _adjust_max_speech_length(self, duration):
        """Adjusts the maximum speech length based on recent speech durations."""
        self.speech_duration_history.append(duration)
        avg_duration = np.mean(self.speech_duration_history)
        self.max_speech_length = max(self.max_speech_length, min(avg_duration * self.adaptive_factor, self.max_max_speech_length))

    def _adjust_speech_timeout(self, pause_duration):
        """Adjusts the speech timeout based on recent pause durations."""
        self.pause_duration_history.append(pause_duration)
        avg_pause = np.mean(self.pause_duration_history)
        self.speech_timeout = min(max(avg_pause * self.adaptive_factor, self.min_speech_timeout), self.max_speech_timeout)

def transcription_callback(transcription, force=False):
    """Handles the transcription output, managing sentence completion and logging."""
    global incomplete_sentence

    full_text = incomplete_sentence + " " + transcription

    # Simple punctuation logic
    sentences = re.split('(?<=[.!?]) +|\n+', full_text)

    for sentence in sentences[:-1]:
        logger.info(f"Transcription: {sentence.strip()}")
        with open("transcriptions.md", "a") as log:
            log.write(f"{sentence.strip()}\n")

    if sentences:
        last_segment = sentences[-1].strip()
        if force or last_segment.endswith(('.', '!', '?')):
            logger.info(f"Transcription: {last_segment}")
            with open("transcriptions.md", "a") as log:
                log.write(f"{last_segment}\n")
            incomplete_sentence = ""
        else:
            incomplete_sentence = last_segment

if __name__ == "__main__":
    recorder = AudioRecorder()
    try:
        recorder.start(transcription_callback)
        logger.info("Transcription started. Speak into your microphone. Press Ctrl+C to stop...")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\nStopping Transcription...")
    finally:
        recorder.stop()
        logger.info("Transcription stopped.")
