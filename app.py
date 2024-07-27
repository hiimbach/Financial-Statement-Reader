import streamlit as st
import json
import pickle
from main import FinStateRead, FORMAT_PROMPT, SUMMARIZE_PROMPT, RAG_PROMPT

st.set_page_config(page_title="Financial Statement Reader")
st.title("Financial Statement Reader")
start = False
reader = None

# Demo option
if st.checkbox("Demo mode"):
    # Use ref_docs and docs from previous run
    with open('ref_docs.pkl', 'rb') as file:
        ref_docs = pickle.load(file)

    with open('documents.pkl', 'rb') as file:
        docs = pickle.load(file)

    reader = FinStateRead(pdf_img_dir='not important anymore',
                                  format_prompt=FORMAT_PROMPT,
                                  summary_prompt=SUMMARIZE_PROMPT,
                                  rag_prompt=RAG_PROMPT,
                                  pre_ref_docs=ref_docs,
                                  pre_docs=docs)
    start = True
else:
    # Upload the PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        # Save the uploaded file
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Split the PDF into images

        reader = FinStateRead(pdf_img_dir='',
                              format_prompt=FORMAT_PROMPT,
                              summary_prompt=SUMMARIZE_PROMPT,
                              rag_prompt=RAG_PROMPT)
        start = True

if start:
    # Main chat interface
    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = reader.run(prompt)
                try:
                    answer = json.loads(response)['answer']
                except:
                    answer = response
                st.write(answer)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)