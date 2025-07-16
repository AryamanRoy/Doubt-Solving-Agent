# Doubt Solving Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a conversational AI assistant focused on physics, leveraging Retrieval-Augmented Generation (RAG) to answer questions based on uploaded PDF documents. The chatbot is built using Streamlit for the user interface and the Groq API for language model inference.

## ✨ Features

-   **PDF Document Ingestion**: Upload PDF files, and the chatbot will extract text content from them.
-   **Text Chunking**: Automatically splits the extracted text into manageable chunks for efficient processing.
-   **Relevant Chunk Retrieval**: Uses TF-IDF and cosine similarity to find the most relevant text chunks from the uploaded PDFs based on user queries.
-   **Groq API Integration**: Utilizes the Groq API (specifically the `llama3-8b-8192` model) to generate concise and accurate answers.
-   **Conversational Context**: Maintains a limited conversational history to provide more coherent responses.
-   **Physics-focused**: Designed to answer physics-related questions and gently redirect users if they ask about other topics.
-   **Streamlit UI**: Provides an intuitive web interface for interacting with the chatbot and uploading documents.

## 📂 Project Structure
    AIDoubtSession/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── app.py
    └── data/
      └── (uploaded_pdfs_will_go_here)


-   `app.py`: The main Streamlit application file that orchestrates the PDF processing, RAG, and chatbot interaction.
-   `requirements.txt`: Lists all the Python dependencies required to run the project.
-   `.gitignore`: Specifies files and directories to be ignored by Git (e.g., virtual environments, data folder, Jupyter checkpoints).
-   `data/`: A directory where uploaded PDF files will be stored.

## 🚀 Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the Repository

bash
git clone https://github.com/your-username/AIDoubtSession.git
cd AIDoubtSession

### 2. Create a Virtual Environment (Recommended)

bash

python -m venv venv

# On Windows

.\venv\Scripts\activate

# On macOS/Linux

source venv/bin/activate

### 3. Install Dependencies

bash

pip install -r requirements.txt

### 4. Obtain a Groq API Key

    Go to the Groq Console.
    Sign up or log in.
    Navigate to the "API Keys" section and create a new API key.
    Replace the placeholder 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 in app.py with your actual Groq API key. Alternatively, and highly recommended for production, set it as an environment variable:

    bash

# On macOS/Linux

export GROQ_API_KEY="your_groq_api_key_here"

# On Windows (Command Prompt)

set GROQ_API_KEY="your_groq_api_key_here"

# On Windows (PowerShell)

    $env:GROQ_API_KEY="your_groq_api_key_here"

    If you use the environment variable, ensure app.py reads from it using os.getenv("GROQ_API_KEY").

### 5. Run the Streamlit Application

bash

streamlit run app.py

This will open the Streamlit application in your web browser.
## 💡 Usage

-   **Upload PDFs**: On the left sidebar of the Streamlit application, use the "Upload PDF documents here" section to upload your physics-related PDF documents. These documents will be used as the knowledge base for the chatbot.
-   **Ask Questions**: Type your physics-related questions into the chat input box at the bottom of the page and press Enter.
-   **Receive Answers**: The chatbot will process your query, retrieve relevant information from the uploaded PDFs, and generate a concise answer.
-   **Maintain Conversation**: The chatbot will remember previous turns in the conversation to provide more contextually relevant responses.

## 🧠 How it Works (RAG Pipeline)

The chatbot employs a Retrieval-Augmented Generation (RAG) architecture:

-   **Document Loading**: PDF files are uploaded by the user, and their text content is extracted using PyPDF2.
-   **Text Chunking**: The extracted text is divided into smaller, fixed-size chunks (e.g., 512 characters). This helps in managing context size and focusing on specific pieces of information.
-   **Retrieval**: When a user asks a question, the system uses TfidfVectorizer and cosine_similarity from scikit-learn to find the top_k (default 3) most relevant text chunks from the entire corpus of uploaded documents. This ensures that only pertinent information is passed to the language model.
-   **Augmentation**: The retrieved relevant chunks, along with the user's query and a system prompt, are combined to form a comprehensive context. This augmented context is then sent to the Groq language model.
-   **Generation**: The Groq llama3-8b-8192 model generates a natural language response based on the provided context and query. The system prompt guides the model to act as a physics expert and to handle non-physics queries appropriately.
-   **Conversational Memory**: A deque (double-ended queue) is used to maintain a limited history of previous assistant responses, which are included in the context for subsequent queries to enable conversational flow.
