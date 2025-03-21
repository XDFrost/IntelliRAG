from enum import Enum
from dataclasses import dataclass
from typing import List, TypedDict, Iterable

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import START

from config.config import Config
from file_loader.file_loader import File

# from data_ingestor.data_ingestor import ingest_files        # InMemory
# from data_ingestor.pinecone_data_ingestor import ingest_files        # Pinecone

import importlib
import streamlit as st
if "db_option" not in st.session_state:
    st.session_state["db_option"] = "InMemory"  
if(st.session_state["db_option"] == "Pinecone"):
    module = importlib.import_module("data_ingestor.pinecone_data_ingestor")
else:
    module = importlib.import_module("data_ingestor.data_ingestor")

importlib.reload(module)        # Reload the module to get the latest changes
ingest_files = getattr(module, "ingest_files")


SYSTEM_PROMPT = """
You're having a conversation with an user about excerpts of their files. Try to be helpful and answer their questions.
If you don't know the answer, you can ask to clarify the question or provide more information.
""".strip()

PROMPT = """
Here's the information you have about the excerpts of the file

<context>
{context}
</context>

One file can have multiple excerpts.

Please respond to the query below

<question>
{question}
</question>

Answer:
"""

FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
""".strip()

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT),
    ]
)


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    role: Role
    content: str

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

@dataclass
class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str


def _remove_thinking_from_message(message: str) -> str:
    close_tag = "</think>"
    tag_length = len(close_tag)
    return message[message.find(close_tag) + tag_length :].strip()

def create_history(Welcome_message: Message) -> List[Message]:
    return [Welcome_message]


class Chatbot:
    def __init__(self, files: List[File]):
        self.files = files
        self.retriever = ingest_files(files)
        self.llm = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            verbose=False,
            keep_alive=-1,
        )
        self.workflow = self._create_workflow()
    
    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            FILE_TEMPLATE.format(name=doc.metadata["source"], content=doc.page_content) 
            for doc in docs
        )
    
    def _retrieve(self, state: State):
        context = self.retriever.invoke(state["question"])
        return {"context": context}
    
    def _generate(self, state: State):
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state["question"],
                "context": self._format_docs(state["context"]),
                "chat_history": state["chat_history"],
            }
        )
        answer = self.llm.invoke(messages)
        return {"answer": answer}
    
    def _create_workflow(self):
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()
    
    def _ask_model(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content)
            for m in chat_history
        ]
        payload = {"question": prompt, "chat_history": history}

        config = {
            "configurable": {"thread_id": 42},
        }
        for event_type, event_data in self.workflow.stream(
            payload, 
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if event_type == "messages":
                chunk, _ = event_data
                yield ChunkEvent(chunk.content)
            if event_type == "updates":
                if "_retrieve" in event_data:
                    documents = event_data["_retrieve"]["context"]
                    yield SourcesEvent(documents)
                if "_generate" in event_data:
                    answer = event_data["_generate"]["answer"]
                    yield FinalAnswerEvent(answer.content)
    
    def ask(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        for event in self._ask_model(prompt, chat_history):
            yield event
            if isinstance(event, FinalAnswerEvent):
                response = _remove_thinking_from_message("".join(event.content))
                # response = "".join(event.content)
                chat_history.append(Message(role=Role.USER, content=prompt))
                chat_history.append(Message(role=Role.ASSISTANT, content=response))
