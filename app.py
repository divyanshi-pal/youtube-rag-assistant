
import streamlit as st
from dotenv import load_dotenv
import os
import time

import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# 1. Get Transcript
def get_transcript(youtube_url):
    try:
        if "youtu.be" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
        elif "v=" in youtube_url:
            video_id = youtube_url.split("v=")[-1].split("&")[0]
        else:
            video_id = youtube_url.strip().split("/")[-1].split("?")[0]

        ytt = YouTubeTranscriptApi()
        
        # Pehle English try karo, phir Hindi
        try:
            fetched = ytt.fetch(video_id, languages=['en', 'en-US', 'en-IN'])
        except:
            fetched = ytt.fetch(video_id, languages=['hi'])
            
        text = " ".join([snippet.text for snippet in fetched])
        return text, video_id
    except Exception as e:
        st.error(f"Transcript Error: {e}")
        return None, None


# 2. Split into chunks
def get_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# 3. Get embeddings using Gemini
def get_embedding(text):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])


# 4. Build vector store (simple numpy-based)
def build_vector_store(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        embeddings.append(emb)
        if i % 5 == 0:
            time.sleep(1)
    return np.array(embeddings), chunks


# 5. Find most relevant chunks
def search_chunks(question, embeddings, chunks, top_k=4):
    question_emb = genai.embed_content(
        model="models/gemini-embedding-001",
        content=question,
        task_type="retrieval_query"
    )
    q_vec = np.array(question_emb['embedding'])
    scores = np.dot(embeddings, q_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-10
    )
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# 6. Generate Summary
def generate_summary(transcript):
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    prompt = f"""
    You are a YouTube video summarizer.
    Summarize the transcript below in clear bullet points within 300 words.
    Cover all important topics discussed.

    Transcript: {transcript}
    """
    response = model.generate_content(prompt)
    return response.text


# 7. Answer Question using RAG
def answer_question(question, embeddings, chunks):
    relevant_chunks = search_chunks(question, embeddings, chunks)
    context = "\n\n".join(relevant_chunks)
    model = genai.GenerativeModel("gemini-3.1-flash-lite")
    prompt = f"""
    You are an expert YouTube video assistant.
    Answer the question using only the context below from the video transcript.
    If answer is not in context, say "This topic was not covered in the video."

    Context:
    {context}

    Question: {question}

    Detailed Answer:
    """
    response = model.generate_content(prompt)
    return response.text


# ─── STREAMLIT UI ───
st.set_page_config(page_title="YouTube RAG Assistant", layout="wide")
st.title("🎥 YouTube RAG-Based Transcript Assistant")
st.markdown("Enter a YouTube link to get **summary** and **ask questions** about the video.")

youtube_url = st.text_input("🔗 Enter YouTube Video URL:")

if youtube_url:
    video_id = youtube_url.strip().split("/")[-1].split("?")[0]
    st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg", width=480)

if st.button("📥 Load Video & Process Transcript"):
    with st.spinner("Fetching transcript and building knowledge base..."):
        transcript, video_id = get_transcript(youtube_url)
        if transcript:
            chunks = get_chunks(transcript)
            embeddings, chunks = build_vector_store(chunks)
            st.session_state["transcript"] = transcript
            st.session_state["chunks"] = chunks
            st.session_state["embeddings"] = embeddings
            st.session_state["processed"] = True
            st.success("✅ Done! Now get summary or ask questions below.")

if st.session_state.get("processed"):
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Get Video Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(st.session_state["transcript"])
                st.markdown("### 📌 Video Summary")
                st.write(summary)

    with col2:
        st.markdown("### ❓ Ask a Question About the Video")
        user_question = st.text_input("Type your question here:")
        if st.button("🔍 Get Answer"):
            if user_question:
                with st.spinner("Finding answer..."):
                    answer = answer_question(
                        user_question,
                        st.session_state["embeddings"],
                        st.session_state["chunks"]
                    )
                    st.markdown("### 💡 Answer")
                    st.write(answer)
            else:
                st.warning("Please enter a question first.")