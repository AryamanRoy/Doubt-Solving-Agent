import os
import time
import streamlit as st
from groq import Client
from PyPDF2 import PdfReader
import glob
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import deque


groq_api_key = "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" # REPLACE WITH YOUR ACTUAL GROQ API KEY
client = Client(api_key=groq_api_key)

def count_tokens(text):
    """Counts the number of tokens in a given text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        # Fallback to word count if tiktoken fails or is not available
        print(f"Tokenization error: {e}. Falling back to word count.")
        return len(text.split())

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or "" # Use 'or ""' to handle pages with no extractable text
    return text

def load_and_extract_texts_from_pdfs(data_folder="data/"):
    """Loads and extracts text from all PDF files in a specified folder."""
    texts = []
    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        st.info(f"Created data directory: {data_folder}")
        return texts # Return empty if no PDFs are there yet

    file_paths = glob.glob(os.path.join(data_folder, "*.pdf"))
    if not file_paths:
        st.info("No PDF files found in the 'data/' directory. Please upload some.")
        return texts

    for file_path in file_paths:
        try:
            text = extract_text_from_pdf(file_path)
            texts.append({"text": text, "file_path": file_path})
        except Exception as e:
            st.error(f"Error processing {file_path}: {e}")
    return texts

def split_text_into_chunks(text, chunk_size=512):
    """Splits a given text into smaller chunks of a specified size."""
    # Ensure text is not empty to avoid errors
    if not text:
        return []
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def retrieve_relevant_chunks(texts, query, top_k=3):
    """Retrieves the most relevant text chunks based on a query using TF-IDF and cosine similarity."""
    if not texts:
        return []

    corpus = [query] + [text["text"] for text in texts]
    
    # Handle cases where corpus might be too small or empty after filtering
    if len(corpus) < 2: # Need at least query and one document text
        return []

    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    
    # Ensure there are enough vectors for similarity calculation
    if vectors.shape[0] < 2:
        return []

    cosine_matrix = cosine_similarity(vectors)
    
    # similarity_scores will be the first row (query similarities) excluding the first element (query to itself)
    similarity_scores = cosine_matrix[0][1:]
    
    # Get indices of top_k similar documents
    # Ensure top_k does not exceed the number of available documents
    actual_top_k = min(top_k, len(similarity_scores))
    ranked_indices = np.argsort(similarity_scores)[-actual_top_k:][::-1] # [::-1] to get in descending order of similarity
    
    relevant_texts = [texts[idx] for idx in ranked_indices]
    return relevant_texts

def generate_response(chunks, query, context_history=None):
    """Generates a response using the Groq API based on relevant chunks and query."""
    system_prompt = "You are here to help the user with their physics inquiries and to respond to greetings. If the user has questions on topics other than physics, you will kindly remind the user to focus on physics-related questions. Provide concise and direct answers."

    # Combine chunks into context
    context_from_chunks = "\n".join(chunks) if chunks else ""

    # Combine context history with current chunks
    full_context = ""
    if context_history:
        full_context += "\n".join(context_history) + "\n"
    full_context += context_from_chunks

    max_allowed_tokens = 5500 # Max tokens for llama3-8b-8192 is 8192, but leaving some buffer
    
    # Check and truncate context if too long
    total_tokens = count_tokens(full_context)
    if total_tokens > max_allowed_tokens:
        # Simple truncation: keep the most recent part of the context
        # This is a basic approach; more sophisticated methods might involve summarization
        # or more intelligent chunk selection.
        full_context = full_context[-(max_allowed_tokens * 4):] # Approx char count for tokens
        total_tokens = count_tokens(full_context) # Recalculate after truncation
        if total_tokens > max_allowed_tokens:
            # If still too large after character-based truncation, return error
            return "⚠️ Context too large even after truncation. Please try a shorter query or fewer documents."

    # Construct messages for the Groq API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {full_context}\nQuery: {query}"}
    ]

    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", # Using the specified model
            max_tokens=400, # Limit response length
            temperature=0.7, # Adjust for creativity vs. directness
            stream=False # Set to True for streaming responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating response: {e}"

def main():
    """Main function for the Streamlit application."""
    st.set_page_config(page_title="Physics Chatbot", page_icon="⚛️")
    st.title("⚛️ Physics Chatbot with PDF-based RAG")
    st.markdown("Upload your physics PDFs and ask questions!")

    # Sidebar for PDF uploads
    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents here",
        accept_multiple_files=True,
        type=["pdf"]
    )

    # Process uploaded files
    if uploaded_files:
        os.makedirs("data", exist_ok=True) # Ensure 'data' directory exists
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            # Check if file already exists to avoid re-writing unnecessarily
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
            else:
                st.sidebar.info(f"File already exists: {uploaded_file.name}")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input for user queries
    if prompt := st.chat_input("Ask a physics-related question:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Load and process PDFs
                texts = load_and_extract_texts_from_pdfs()
                
                # Retrieve relevant chunks based on the current query
                relevant_texts = retrieve_relevant_chunks(texts, prompt)
                
                # Extract text content from relevant_texts for chunking
                relevant_text_content = [item["text"] for item in relevant_texts]
                
                # Split relevant text content into chunks
                chunks = []
                for text_content in relevant_text_content:
                    chunks.extend(split_text_into_chunks(text_content))

                # Prepare context history for the LLM
                # Only include assistant's previous responses to avoid self-referencing user queries
                context_history_content = [m["content"] for m in st.session_state.messages if m["role"] == "assistant"]

                # Generate response using the RAG pipeline
                response = generate_response(chunks, prompt, context_history_content)
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
