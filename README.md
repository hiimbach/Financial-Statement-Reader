# Financial Statement Reader
Financial Statement Chatbot is a Retrieval-Augmented Generation (RAG) system designed to interact with financial 
statement data. This project leverages OCR, summarization, and embedding techniques to enable users to query and 
retrieve information from financial statements effectively.

## About the Project
Key features include:

- Converting PDF OCR files to JSON using Tesseract.
- Summarizing PDFs for efficient data retrieval.
- Utilizing sentence-transformers/all-MiniLM-L6-v2 as an embedder for RAG.
- A user-friendly interface built with Streamlit.

## Quick Start with Docker
Run the following command to build the Docker image:
```
docker build -t finread . 
```

Run the following command to start the Docker container:
```
docker run -p 8501:8501 finread
```

## Installation
Run the following commands to install the required packages:

### Install Tesseract in Brew with Vietnamese language
```
brew install tesseract
brew install tesseract-lang
```

### Install Poppler in Brew
```
brew install poppler
```

### Install Python packages
```
pip install -r requirements.txt
```

## Usage
Run the following command to start the Streamlit app:
```
streamlit run app.py
```
In the app you can choose Demo if you dont have a financial statement file to upload.

## Contact 
If you have any questions, feel free to reach out to me at: 
- Email: lenhobach@gmail.com
- GitHub: hiimbach
