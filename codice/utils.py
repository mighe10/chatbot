from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from transformers import AutoTokenizer
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

class TxtFile:
    def __init__(self, directory, name_file):
        file_directory = os.path.join(directory, name_file)
        print(file_directory)
        #file = open(file_directory, 'r')
        self.page_content = ""
        
        try:
            with open(file_directory, 'r', encoding='utf-8') as file:
                lines=file.readlines()
        except UnicodeDecodeError:
            with open(file_directory, 'r', encoding='latin1') as file:
                lines=file.readlines()
                
        #lines = file.readlines()
        for line in lines:
            self.page_content += (line+" ")
        self.metadata = {
            "source": file_directory
        }
        
def txt_loader():
    txt_directory = os.path.join(os.path.dirname(__file__), "RegTXT")
    print(txt_directory)
    pages = []
    
    if(os.path.isdir(txt_directory)):
        txt_files = [f for f in os.listdir(txt_directory) if f.endswith('.txt')]
        for file in txt_files:
            pages.append(TxtFile(txt_directory, file))        
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ),
        chunk_size = 264,
        chunk_overlap=32,
        strip_whitespace=True,
    )
    
    docs = text_splitter.split_documents(pages)
    return docs
    
class Encoder:
    def __init__(self, model_name: str, device):
        #questo modello crea un dense vector store 384 dimensionale
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    
def prepare_rag_llm(
     loaded_db
):
    
    llm=Ollama(model='llama3')

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation


def generate_answer(question):
    response = st.session_state.conversation({"question": question})
    answer = response.get("answer").split("Helpful Answer:")[-1].strip()
    explanation = response.get("source_documents", [])
    doc_source = [d for d in explanation]

    return answer, doc_source
    