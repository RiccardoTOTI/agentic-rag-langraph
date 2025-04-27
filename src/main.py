# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from retrieval_graph.graph import graph
from retrieval_graph.state import InputState, State
from retrieval_graph.elasticsearch_client import get_elasticsearch_chat_message_history, log_event_to_elasticsearch
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Request models
class MessageInput(BaseModel):
    content: str

class ConversationInput(BaseModel):
    messages: List[MessageInput]
    session_id: Optional[str] = None

# Response models
class RetrievedDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    response: str
    documents: List[RetrievedDocument]
    retrieval_used: bool
    session_id: str

# Utilizziamo la funzione già esistente nel modulo elasticsearch_client

@app.post("/chat", response_model=ChatResponse)
async def chat(input_data: ConversationInput):
    try:
        messages = [HumanMessage(content=msg.content) for msg in input_data.messages]
        input_state = InputState(messages=messages)
        
        # Usa la session_id dalla richiesta o genera una nuova con uuid
        session_id = input_data.session_id or str(uuid.uuid4())
        
        # Crea un RunnableConfig con la session_id e altre configurazioni
        config = RunnableConfig(
            configurable={
                "session_id": session_id,
                "es_index_chat_history": os.getenv("ES_INDEX_CHAT_HISTORY", "workplace-app-docs-chat-history")
            }
        )

        # Esecuzione del grafo
        result = await graph.ainvoke(input_state, config=config)

        # Accesso corretto agli output
        response_messages = result.get("messages", [])
        documents = result.get("retrieved_docs", [])
        last_message = response_messages[-1].content if response_messages else ""

        # Salva la conversazione in Elasticsearch
        if response_messages:
            chat_history = get_elasticsearch_chat_message_history(
                os.getenv("ES_INDEX_CHAT_HISTORY", "workplace-app-docs-chat-history"),
                session_id
            )
            
            # Aggiungi l'ultimo messaggio utente e la risposta AI
            last_user_message = messages[-1].content if messages else ""
            if last_user_message:
                chat_history.add_user_message(last_user_message)
            
            chat_history.add_ai_message(last_message)
            
            # Registra eventi di logging
            log_event_to_elasticsearch(
                session_id, 
                "chat_completion", 
                {
                    "user_query": last_user_message,
                    "ai_response": last_message,
                    "retrieval_used": bool(documents)
                }
            )

        return ChatResponse(
            response=last_message,
            documents=[
                RetrievedDocument(content=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ],
            retrieval_used=bool(documents),
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))