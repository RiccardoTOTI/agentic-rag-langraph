"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast, Optional
import os

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from retrieval_graph import retrieval
from retrieval_graph.configuration import Configuration
from retrieval_graph.state import InputState, State
from retrieval_graph.utils import format_docs, get_message_text, load_chat_model
from retrieval_graph.elasticsearch_client import get_elasticsearch_chat_message_history, log_event_to_elasticsearch

# Define the function that calls the model
class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""
    query: str

class ShouldRetrieveDecision(BaseModel):
    """Structured output for deciding whether to retrieve documents."""
    should_retrieve: bool

# Utilizziamo la funzione già esistente nel modulo elasticsearch_client

def get_chat_history_from_es(config: RunnableConfig) -> list[BaseMessage]:
    """Retrieve chat history from Elasticsearch if a session_id is provided."""
    configurable = config.get("configurable", {})
    session_id = configurable.get("session_id")
    es_index = configurable.get("es_index_chat_history", "workplace-app-docs-chat-history")
    
    if not session_id:
        return []
    
    chat_history = get_elasticsearch_chat_message_history(es_index, session_id)
    return chat_history.messages

async def should_retrieve(
    state: State, *, config: RunnableConfig
) -> str:
    """
    Decide whether to perform retrieval or not using an LLM.
    Returns 'use_retrieval' or 'skip_retrieval'.
    """
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", configuration.retrieve_decision_prompt),
        ("human", "Messaggio utente:\n{query}")
    ])

    model = load_chat_model(configuration.query_model).with_structured_output(
        ShouldRetrieveDecision
    )

    query = state.queries[-1]
    prompt_input = {"query": query}
    message = await prompt.ainvoke(prompt_input, config)
    decision = await model.ainvoke(message, config)
    
    # Log decision event if session_id is available
    configurable = config.get("configurable", {})
    session_id = configurable.get("session_id")
    if session_id:
        log_event_to_elasticsearch(
            session_id,
            "retrieval_decision",
            {"should_retrieve": decision.should_retrieve, "query": query}
        )

    return "use_retrieval" if decision.should_retrieve else "skip_retrieval"

async def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration."""
    messages = state.messages
    
    # Recupera la cronologia delle chat da Elasticsearch se esiste una session_id
    es_history = get_chat_history_from_es(config)
    
    if len(messages) == 1 and not es_history:
        # È la prima domanda dell'utente senza cronologia. Usa l'input direttamente.
        human_input = get_message_text(messages[-1])
        return {"queries": [human_input]}
    else:
        configuration = Configuration.from_runnable_config(config)
        
        # Combina i messaggi attuali con la cronologia di Elasticsearch
        combined_messages = es_history + messages
        
        # Crea il prompt per generare la query
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", configuration.query_system_prompt),
                ("placeholder", "{messages}"),
            ]
        )
        model = load_chat_model(configuration.query_model).with_structured_output(
            SearchQuery
        )

        message_value = await prompt.ainvoke(
            {
                "messages": combined_messages,
                "queries": "\n- ".join(state.queries),
                "system_time": datetime.now(tz=timezone.utc).isoformat(),
            },
            config,
        )
        generated = cast(SearchQuery, await model.ainvoke(message_value, config))
        return {
            "queries": [generated.query],
        }

async def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state."""
    with retrieval.make_retriever(config) as retriever:
        query = state.queries[-1]
        response = await retriever.ainvoke(query, config)
        
        # Log retrieval event if session_id is available
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        if session_id:
            log_event_to_elasticsearch(
                session_id,
                "document_retrieval",
                {
                    "query": query,
                    "doc_count": len(response),
                    "doc_ids": [doc.metadata.get("id", "unknown") for doc in response]
                }
            )
            
        return {"retrieved_docs": response}

async def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our 'agent'."""
    configuration = Configuration.from_runnable_config(config)
    
    # Recupera la cronologia delle chat se esiste una session_id
    es_history = get_chat_history_from_es(config)
    
    # Combina i messaggi attuali con la cronologia di Elasticsearch
    combined_messages = es_history + state.messages

    has_docs = bool(state.retrieved_docs)
    system_prompt = (
        configuration.response_system_prompt if has_docs
        else configuration.direct_answer_system_prompt
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{messages}"),
        ]
    )

    model = load_chat_model(configuration.response_model)
    retrieved_docs = format_docs(state.retrieved_docs) if has_docs else ""

    message_value = await prompt.ainvoke(
        {
            "messages": combined_messages,
            "retrieved_docs": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = await model.ainvoke(message_value, config)
    
    # Log response event if session_id is available
    configurable = config.get("configurable", {})
    session_id = configurable.get("session_id")
    if session_id:
        log_event_to_elasticsearch(
            session_id,
            "ai_response",
            {
                "response_length": len(response.content) if hasattr(response, "content") else 0,
                "used_retrieved_docs": has_docs
            }
        )
    
    return {"messages": [response]}

# Define the graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

builder.add_node("generate", generate_query)
builder.add_node("retrieve", retrieve)
builder.add_node("respond", respond)

builder.add_edge("__start__", "generate")
builder.add_conditional_edges(
    "generate",           # ← nodo di partenza
    should_retrieve,      # ← funzione che ritorna 'use_retrieval' o 'skip_retrieval'
    {
        "use_retrieval": "retrieve",
        "skip_retrieval": "respond"
    }
)

builder.add_edge("retrieve", "respond")

graph = builder.compile(
    interrupt_before=[],
    interrupt_after=[],
)
graph.name = "RetrievalGraph"