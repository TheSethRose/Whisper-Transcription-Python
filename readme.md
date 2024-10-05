# Real-Time Adaptive Speech Transcription (Proof of Concept)

## Overview

This project demonstrates a real-time, adaptive speech transcription system using OpenAI's Whisper Turbo model. It captures audio from a microphone, dynamically adjusts to the speaker's patterns, and outputs transcriptions in real-time. This proof of concept showcases the capabilities of the `whisper-large-v3-turbo` model in a practical application.

## Key Features

- **Real-Time Processing**: Continuously captures and transcribes audio input.
- **State-of-the-Art Model**: Utilizes OpenAI's Whisper Turbo for accurate speech recognition.
- **Adaptive Speech Detection**: Dynamically adjusts to the speaker's pace and pausing patterns.
- **Concurrent Operation**: Uses multi-threading for simultaneous audio capture and transcription.
- **Efficient Buffering**: Implements an adaptive audio buffer for optimal processing.
- **GPU Acceleration**: Leverages CUDA for enhanced performance on compatible hardware.
- **Long Speech Handling**: Capable of processing extended, uninterrupted speech without artificial splits.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for improved performance)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-time-speech-transcription-poc.git
   cd real-time-speech-transcription-poc
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script:

```
python main.py
```

Speak into your microphone. The system will adapt to your speaking patterns and output transcriptions in real-time. Press Ctrl+C to stop the process.

## Implementation Details

- `TranscriptionService`: Manages the Whisper Turbo model and transcription process.
- `AudioRecorder`: Handles continuous audio capture, buffering, and adaptive speech detection.
- Adaptive features dynamically adjust maximum speech length and timeout based on the speaker's patterns.
- Implements logic to handle incomplete sentences and very long utterances.

## Limitations and Considerations

- Transcription quality depends on audio input quality and environmental factors.
- Resource-intensive; performs best with GPU acceleration.
- Currently optimized for English language transcription.
- For extremely long, uninterrupted speech (>2 minutes), transcriptions may be output in chunks while maintaining context.

## Contributing

This is a proof of concept, but contributions are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper Turbo model
- Hugging Face for the Transformers library
- The PyAudio development team

---

**Note**: This is a proof of concept and may require further development and optimization for production use.
