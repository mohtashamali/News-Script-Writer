# extract.py
import pdfplumber
import docx
from youtube_transcript_api import YouTubeTranscriptApi

# --------------------------
# PDF Extraction
# --------------------------
def extract_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
    return text

# --------------------------
# TXT Extraction
# --------------------------
def extract_txt(file_path):
    text = ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"Error extracting TXT: {e}")
    return text

# --------------------------
# DOCX Extraction
# --------------------------
def extract_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX: {e}")
    return text

# --------------------------
# YouTube Transcript Extraction
# --------------------------
def extract_youtube(video_id):
    text = ""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t['text'] for t in transcript])
    except Exception as e:
        print(f"Error extracting YouTube transcript: {e}")
    return text

# --------------------------
# Clean Extracted Text
# --------------------------
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()
