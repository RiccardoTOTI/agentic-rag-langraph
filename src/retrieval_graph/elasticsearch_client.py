import os
from dotenv import load_dotenv


from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchChatMessageHistory
from datetime import datetime

load_dotenv()


if os.environ.get("ELASTICSEARCH_USER"):
    elasticsearch_client = Elasticsearch(
        hosts=[os.environ["ELASTICSEARCH_URL"]],
        basic_auth=(os.environ["ELASTICSEARCH_USER"], os.environ["ELASTICSEARCH_PASSWORD"]),
    )
else:
    raise ValueError(
            "Please provide ELASTICSEARCH URL, USER OR PASSWORD"
    )

def get_elasticsearch_chat_message_history(index, session_id):
    return ElasticsearchChatMessageHistory(
        es_connection=elasticsearch_client, index=index, session_id=session_id
    )

def log_event_to_elasticsearch(session_id, event_type, payload):
    """
    Log an event to Elasticsearch for observability.

    Args:
        session_id (str): Unique session identifier.
        event_type (str): Type of event (e.g. 'rag_decision', 'retrieved_doc', 'answer').
        payload (dict): Arbitrary data to store.
    """
    try:
        index = os.getenv("ES_INDEX_EVENT_LOG", "workplace-app-event-log")

        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "payload": payload,
        }

        elasticsearch_client.index(index=index, document=doc)
    except Exception as e:
        if hasattr(current_app, "logger"):
            current_app.logger.warning(f"Failed to log event to Elasticsearch: {e}")
        else:
            print(f"[WARN] Failed to log event: {e}")
