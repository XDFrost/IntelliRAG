import random
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from chatbot.chatbot import Chatbot, ChunkEvent, Message, Role, SourcesEvent, create_history
from file_loader.file_loader import load_uploaded_file


LOADING_MESSAGES = [
    "Hold on, I'm wrestling with some digital goblins... literally.",
    "Loading.. please try not to panic, I'm a professional.",
    "I'm working on it, I swear.",
    "Just a moment, I'm busy saving the sapce-time continuum. Oops, I've said too much.",
    "Please withhold your judgement, I'm doing my best.",
    "Hold your horses, I'm busy turning coffee into code.",
    "Loading... I'm not a wizard, but I'm working on it.",
    "Hold on, while I figure out if your request is a cosmic joke.",
    "Hang tight, I'm busy tickling the algorithms.",
    "Loading... Might as well grab a coffee while you wait.",
    "Hold on, I'm busy turning 1s and 0s into magic.",
    "Just a sec, wrangling the data pixies.",
    "Loading... I'm not a robot, but I'm working on it.",
    "Processing... Can't rush perfection.",
    "Processing... beacause apparently some digital miracles take time.",
]

WELCOME_MESSAGE = Message(role=Role.ASSISTANT, content="Hello, I'm your AI assistant IntelliRAG. How can I help you today?")

st.set_page_config(
    page_title="IntelliRAG",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("IntelliRAG: AI Assistant for Document Analysis and web scraping")
st.subheader("Private intelligence for your thoughts and files")

@st.cache_resource(show_spinner=False)
def create_chatbot(files: List[UploadedFile]):
    files = [load_uploaded_file(file) for file in files]
    return Chatbot(files)

def show_uploaded_documents() -> List[UploadedFile]:
    holder = st.empty()
    with holder.container():
        uploaded_files = st.file_uploader(
            label="Upload PDF, MD, or TXT files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
        )
    if not uploaded_files:
        st.warning("Please upload a file to continue!")
        st.stop()
    
    with st.spinner("Analyzing your files..."):
        holder.empty()
        return uploaded_files
    
uploaded_files = show_uploaded_documents()
chatbot = create_chatbot(uploaded_files)

if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

with st.sidebar:
    st.title("📂 Document Repository")
    st.divider()
    
    if chatbot.files:
        for i, file in enumerate(chatbot.files, start=1):
            file_extension = file.name.split('.')[-1].lower()
            
            # File type icon based on extension
            if file_extension == 'pdf':
                file_icon = "📄"
            elif file_extension == 'md':
                file_icon = "📝"
            elif file_extension == 'txt':
                file_icon = "📃"
            else:
                file_icon = "📎"
            
            # Create a card-like container for each file
            st.markdown(
                f"""
                <div style='padding: 12px 15px; 
                           margin-bottom: 10px; 
                           background-color: rgba(240, 242, 246, 0.4); 
                           border-radius: 8px; 
                           border-left: 4px solid #4E8DF5;
                           box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>
                    <div style='display: flex; align-items: center;'>
                        <span style='font-size: 1.2rem; margin-right: 10px;'>{file_icon}</span>
                        <span style='font-weight: 500;'>{file.name}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Add additional sidebar information
    st.divider()
    st.caption("Supported formats: PDF, Markdown, TXT")
    
    # Add a mini stats section 
    if chatbot.files:
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Files", len(chatbot.files))
        with col2:
            st.metric("Types", len(set(f.name.split('.')[-1].lower() for f in chatbot.files)))

for message in st.session_state.messages:
    avatar = "🤖" if message.role == Role.ASSISTANT else "🧑‍💼"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Type your messages..."):
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                for i, doc in enumerate(event.content):
                    with st.expander(f"Source #{i+1}"):
                        st.markdown(doc.page_content)
            if isinstance(event, ChunkEvent):
                chunk = event.content
                full_response += chunk
                message_placeholder.markdown(full_response)
