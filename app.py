import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables at the very start
load_dotenv()

from scriptmaker import create_db, generate_news
from extract import extract_docx, extract_pdf, extract_txt, extract_youtube, clean_text

# -----------------------------------------------
# STREAMLIT APP UI
# -----------------------------------------------

st.set_page_config(page_title="RAG News Script Generator", layout="wide")

st.title("📰 RAG-Based News Script Generator")
st.write("Upload files, paste text, or enter a YouTube link — and generate a clean, ready-to-read news script.")

st.divider()

# Input section
col1, col2 = st.columns(2)

uploaded_file = col1.file_uploader("Upload PDF / TXT / DOCX", type=["pdf", "txt", "docx"])
youtube_link = col2.text_input("Or enter a YouTube video link")

manual_text = st.text_area(
    "Or paste raw text here (optional)", 
    height=200,
    placeholder="Paste any article, speech, paragraph, story, or raw text..."
)

run_btn = st.button("Generate News Script")


# -----------------------------------------------
# PROCESSING BLOCK
# -----------------------------------------------

if run_btn:
    if not uploaded_file and not youtube_link and not manual_text.strip():
        st.error("Please upload a file, paste text, or enter a YouTube link.")
        st.stop()

    # ============================================
    # STEP 1: TEXT EXTRACTION
    # ============================================
    with st.spinner("📄 Extracting and preparing text..."):

        extracted_text = ""

        # File uploaded
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            if uploaded_file.type == "application/pdf":
                extracted_text = extract_pdf(temp_path)

            elif uploaded_file.type == "text/plain":
                extracted_text = extract_txt(temp_path)

            else:  # DOCX
                extracted_text = extract_docx(temp_path)

            os.remove(temp_path)

        # YouTube link
        elif youtube_link:
            try:
                video_id = youtube_link.split("v=")[-1].split("&")[0]
                extracted_text = extract_youtube(video_id)
            except Exception as e:
                st.error(f"Could not fetch transcript: {str(e)}")
                st.stop()

        # ⭐ Manual pasted text (YOUR CASE)
        elif manual_text and len(manual_text.strip()) > 0:
            extracted_text = manual_text.strip()
            st.info(f"✅ Extracted {len(extracted_text)} characters from pasted text")

        # Clean text
        extracted_text = clean_text(extracted_text)

        if len(extracted_text) < 50:
            st.error("Not enough text found. Please try a better input.")
            st.stop()

        st.success(f"✅ Text extracted successfully! ({len(extracted_text)} characters)")

    # ============================================
    # STEP 2: CREATE VECTOR DATABASE 🔥
    # ============================================
    with st.spinner("🧠 Creating Vector Database from your text..."):
        try:
            # 🎯 THIS IS WHERE THE MAGIC HAPPENS
            # Your pasted text gets:
            # 1. Split into chunks (800 chars each, 150 overlap)
            # 2. Converted to embeddings using BGE model
            # 3. Stored in FAISS vector database
            
            db = create_db(extracted_text)
            
            # Show what happened
            st.success("✅ Vector Database created successfully!")
            
            # Optional: Show database stats
            with st.expander("📊 Vector Database Stats"):
                st.write(f"**Total text length:** {len(extracted_text)} characters")
                st.write(f"**Estimated chunks:** ~{len(extracted_text) // 800} chunks")
                st.write(f"**Embedding model:** BAAI/bge-small-en-v1.5")
                st.write(f"**Vector store:** FAISS")
                
        except Exception as e:
            st.error(f"❌ Error creating vector database: {str(e)}")
            st.stop()

    # ============================================
    # STEP 3: GENERATE NEWS SCRIPT USING RAG
    # ============================================
    with st.spinner("📝 Generating news script from vector database..."):
        try:
            # This function:
            # 1. Searches the vector DB for relevant chunks
            # 2. Retrieves top 6 most relevant chunks
            # 3. Sends them to LLM to generate script
            
            final_script = generate_news(db)
            
        except Exception as e:
            st.error(f"❌ Error generating script: {str(e)}")
            st.stop()

    # ============================================
    # STEP 4: DISPLAY RESULTS
    # ============================================
    st.success("🎉 News script generated successfully!")
    st.subheader("📄 Final News Script:")
    st.write(final_script)

    st.download_button(
        label="⬇ Download Script as TXT",
        data=final_script,
        file_name="news_script.txt",
        mime="text/plain"
    )