import streamlit as st
import utils
import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import gc
from PIL import Image
from pathlib import Path
import base64

def set_background():
    page_bg_img = """
    <style>
    .stApp {
    background-image: url('app/static/sfondo_napoli.png');
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
def chat_history():
    chats_path = os.path.join(os.path.dirname(__file__), "chat_history")
    chats = [
        "Nuova chat"
        ]
    for chat in os.listdir(chats_path):
        chat = chat[:-len(".txt")]
        chats.append(chat)
    return chats

def open_chat_from_file(name_file):
    history = []
    new_chat = name_file == "Nuova chat"
    if st.session_state.overwriting and not new_chat:
        with open("chat_history/"+name_file+".txt", 'r') as file:
            lines = file.readlines()
            lines_range = len(lines)
        for i in range(lines_range):
            if lines[i] == "user\n":
                k=i+1
                user_content = ""
                while k<lines_range and lines[k] != "assistant\n":
                    user_content = user_content + lines[k]
                    k=k+1
                history.append({"role": "user", "content": user_content})
            elif lines[i] == "assistant\n":
                k=i+1
                assistant_content = ""
                while k<lines_range and lines[k] != "user\n":
                    assistant_content = assistant_content + lines[k]
                    k=k+1
                history.append({"role": "assistant", "content": assistant_content})
    elif not st.session_state.overwriting:
        history = st.session_state.history
    return history

def write_on_history_file(name_file, question, answer):
    file = open("chat_history/"+name_file+".txt", 'a')
    file.write("user\n")
    file.write(question+"\n")
    file.write("assistant\n")
    file.write(answer+"\n")
    file.close()
    
def get_base64_of_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return base64_pdf

def pdf_download_link(pdf_path,name_file):
    base64_pdf = get_base64_of_pdf(pdf_path)
    st.link_button(name_file, f"data:application/pdf;base64,{base64_pdf}")

def main():
    clear_gpu_memory()
    im = Image.open('images/teschio.png')
    st.set_page_config(page_title="One site", page_icon=im)
    set_background()
    encoder = utils.Encoder("sentence-transformers/all-MiniLM-L12-v2","cpu")
    document_embedding(encoder.embedding_function)
    display_chat_modificato(encoder.embedding_function)
        
def document_embedding(encoder):
    docs = utils.txt_loader()
    if not os.path.isdir(os.path.join(os.path.dirname(__file__), "vector store/CameraDeputati")):
        print("La directory non esiste")
        faiss_db = FAISS.from_documents(docs, encoder, distance_strategy=DistanceStrategy.COSINE)
        faiss_db.save_local("vector store/CameraDeputati")
    else:
        print("La directory esiste")

def display_chat_modificato(encoder):
    
    loaded_db = FAISS.load_local(
        "vector store/CameraDeputati", encoder, allow_dangerous_deserialization=True
    )
    
    user_image = Image.open('images/teschio.png')
    assistant_image = Image.open('images/internet-bot-computer-icons-chatbot-clip-art-sticker-47add9a9c3c071ac5a836f6bb8243772.png')
    st.title("FORZA NAPOLI SEMPRE (FNS)")
    st.sidebar.title("CHAT HISTORY: ")
    history_list = chat_history()
    selection = st.sidebar.radio("GO TO: ", 
                                 history_list
                                 )
        
    # Prepare the LLM model
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.session_state.conversation = utils.prepare_rag_llm(
        loaded_db
    )
    
    if "overwriting" not in st.session_state:
        st.session_state.overwriting = True
        
    if "not_saved_chat" not in st.session_state:
        st.session_state.not_saved_chat = False
    
    # Chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    if not st.session_state.not_saved_chat:
        st.session_state.overwriting = True
    else:
        if selection == "Nuova chat" and st.session_state.not_saved_chat:
            st.session_state.overwriting = False
        else:
            st.session_state.overwriting = True
            st.session_state.not_saved_chat = False
    st.session_state.history = open_chat_from_file(selection)

    # Source documents
    if "source" not in st.session_state:
        st.session_state.source = []
    
    # Display chats
    for message in st.session_state.history:
        if message["role"] == "user":
            image = user_image
        else:
            image = assistant_image
        with st.chat_message(message["role"], avatar = image):
            st.markdown(message["content"])

    # Ask a question
    if question := st.chat_input("Ask a question"):
        # Append user question to history
        st.session_state.history.append({"role": "user", "content": question})
        # Add user question
        with st.chat_message("user", avatar=user_image):
            st.markdown(question)

        # Answer the question
        answer, doc_source = utils.generate_answer(question)
        with st.chat_message("assistant", avatar=assistant_image):
            st.write(answer)
        if(selection == "Nuova chat"):
            st.session_state.not_saved_chat = True
        else:
            st.session_state.not_saved_chat = False
            write_on_history_file(selection, question, answer)
        # Append assistant answer to history
        st.session_state.history.append({"role": "assistant", "content": answer})
        
        # Create hypertestual links
        for source in doc_source:
            source_name = Path(source.metadata["source"][:-4]).name
            source_directory = os.path.join(os.path.dirname(__file__), "TextMiningPdf")
            source_directory = os.path.join(source_directory,source_name+".pdf")
            pdf_download_link(source_directory,source_name)
        # Append the document sources
        st.session_state.source.append({"question": question, "answer": answer, "document": doc_source})


    # Source documents
    with st.expander("Chat History and Source Information"):
        st.write(st.session_state.source)

        
if __name__ == "__main__":
    main()