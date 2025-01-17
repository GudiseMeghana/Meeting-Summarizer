import os
import smtplib
from email.message import EmailMessage
import moviepy.editor as mp
import whisper
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from pyAudioAnalysis import audioSegmentation as aS
import streamlit as st
import time

# Custom CSS to set background image and style elements
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.veeforu.com/wp-content/uploads/2022/10/Blue-pastel-gradient-background-free-download.-scaled.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stText {
        background-color: rgba(255, 255, 255, 0.3);
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
    }
    .stButton>button {
        background-color: #008CBA;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stText p {
        font-size: 18px;
        font-weight: 500;
    }
    .stMarkdown p {
        font-size: 16px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 12px;
        color: #888;
    }
    </style>
    """, unsafe_allow_html=True
)

# Convert Video to Audio
def video_to_audio(video_file, audio_file):
    try:
        video = mp.VideoFileClip(video_file)
        video.audio.write_audiofile(audio_file)
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

# Transcription using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]

# Advanced Sentiment Analysis
def sentiment_analysis(text, chunk_size=500):
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    positive_count, negative_count = 0, 0
    
    for chunk in text_chunks:
        result = sentiment_analyzer(chunk)[0]
        if result['label'] == 'POSITIVE':
            positive_count += 1
        else:
            negative_count += 1
    
    total = len(text_chunks)
    positive_percentage = (positive_count / total) * 100
    negative_percentage = (negative_count / total) * 100
    neutral_percentage = 100 - (positive_percentage + negative_percentage)
    overall_sentiment = "POSITIVE" if positive_percentage > negative_percentage else "NEGATIVE"
    
    return overall_sentiment

# Speaker Diarization
def speaker_diarization(audio_file):
    [flags, classes, centers] = aS.speaker_diarization(audio_file, n_speakers=4)
    return len(set(flags))

# Generate Summary using Transformers
def generate_summary(text):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to("cpu")
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=500,
        min_length=200,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Generate Plan of Action using Transformers
def generate_plan_of_action(text):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    prompt = "Extract the detailed action points from the following meeting transcript:"
    action_input = prompt + text
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to("cpu")
    inputs = tokenizer([action_input], max_length=1024, return_tensors="pt", truncation=True)
    action_ids = model.generate(
        inputs["input_ids"],
        max_length=200,
        min_length=100,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(action_ids[0], skip_special_tokens=True)

# Detect Genre
def classify_genre(text):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        genres = ["Product", "Marketing", "Technical", "General","Finance"]
        tokenizer = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-CoLA")
        model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-CoLA").to(device)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_genre_idx = torch.argmax(probs, dim=1).item()
        return genres[predicted_genre_idx]
    except Exception as e:
        st.error(f"Error classifying genre: {e}")
        return "Unknown"

# Send Email
def send_email(receiver_email, subject, body, sender_email, app_password):
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Streamlit UI
st.title("Meeting Summarization and Plan of Action Generator")

uploaded_file = st.file_uploader("Upload meeting video file", type=["mp4"], help="Upload the video file for analysis.")
progress_bar = st.progress(0)

# Email details
sender_email = "yourmail@gmail.com"
app_password = "apppassword"
receiver_email = st.text_input("Receiver's Email", help="Enter the email to send the analysis to.")

# File Size Limitation
MAX_FILE_SIZE = 50 * 1024 * 1024
if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.warning("The uploaded file is too large. Please upload a video under 50MB.")

# Process video and generate analysis
if st.button("Summarize and Send Email") and uploaded_file:
    with st.spinner("Processing video..."):
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)

        audio_file = "audio.wav"
        if video_to_audio(video_path, audio_file):
            transcription = transcribe_audio(audio_file)

            if transcription:
                with st.expander("Summary"):
                    summary = generate_summary(transcription)
                    st.markdown(f'<div class="stText">{summary}</div>', unsafe_allow_html=True)

                with st.expander("Genre"):
                    genre = detect_genre(transcription)
                    st.markdown(f'<div class="stText">{genre}</div>', unsafe_allow_html=True)

                with st.expander("Overall Sentiment"):
                    sentiment = sentiment_analysis(transcription)
                    st.markdown(f'<div class="stText">{sentiment}</div>', unsafe_allow_html=True)

                with st.expander("Number of Speakers"):
                    num_speakers = speaker_diarization(audio_file)
                    st.markdown(f'<div class="stText">{num_speakers}</div>', unsafe_allow_html=True)

                with st.expander("Plan of Action"):
                    action_points = generate_plan_of_action(transcription)
                    st.markdown(f'<div class="stText">{action_points}</div>', unsafe_allow_html=True)

                email_content = f"Meeting Summary:\n{summary}\n\nGenre: {genre}\n\nOverall Sentiment: {sentiment}\n\nSpeakers: {num_speakers}\n\nPlan of Action:\n{action_points}"
                email_sent = send_email(receiver_email, "Meeting Analysis Results", email_content, sender_email, app_password)

                if email_sent:
                    st.success("Email sent successfully!")

        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_file):
            os.remove(audio_file)
