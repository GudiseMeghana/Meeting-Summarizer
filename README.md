# Meeting Summarizer 

This repository contains a Python application that provides an intuitive user interface (UI) for summarizing meeting recordings. It leverages advanced machine learning models and tools to process audio/video files, generate transcripts, and summarize the content efficiently.

## Features

- **Audio/Video Processing**: Handles audio and video inputs for meeting recordings.
- **Transcription**: Converts speech to text using OpenAI Whisper.
- **Text Summarization**: Summarizes the transcriptions using Hugging Face Transformers (BART model).
- **Speaker Diarization**: Identifies and labels different speakers in the audio.
- **Streamlit UI**: User-friendly web-based interface for easy interaction.

## Technologies Used

- **Streamlit**: For creating the web-based UI.
- **Whisper**: For high-quality audio transcription.
- **Hugging Face Transformers**: For text summarization using the BART model.
- **MoviePy**: For handling video processing.
- **PyAudioAnalysis**: For speaker diarization.

## Installation

Follow the steps below to set up and run the application:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/meeting-summarizer-ui.git
   cd meeting-summarizer-ui
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required Whisper model (if not already downloaded):
   ```python
   import whisper
   model = whisper.load_model("base")  # Change "base" to your preferred model size
   ```

4. Run the application:
   ```bash
   streamlit run meetsummariserwithui.py
   ```

## Usage

1. Open the Streamlit app in your browser (default: `http://localhost:8501`).
2. Upload a meeting recording (audio/video file).
3. Select the processing options (e.g., speaker diarization, summarization).
4. View the transcription and summarized notes on the UI.
5. Download or share the results as needed.

## File Structure

- `meetsummariserwithui.py`: Main script containing the application logic.
- `requirements.txt`: List of required Python libraries.
- `assets/`: Directory for UI assets like images and CSS files (if any).

## Requirements

- Python 3.8 or higher
- Libraries listed in `requirements.txt`

## Key Libraries in `requirements.txt`

```text
streamlit
openai-whisper
transformers
moviepy
pyAudioAnalysis
```

## Contributing

Contributions are welcome! If you have ideas for improvements or new features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch for your feature: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- OpenAI for Whisper
- Hugging Face for Transformers
- Streamlit for simplifying UI development
- PyAudioAnalysis and MoviePy for audio/video handling

