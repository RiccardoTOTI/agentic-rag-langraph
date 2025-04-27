import os
from dotenv import load_dotenv
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig, ensure_config
from retrieval_graph import prompts


load_dotenv()


class IndexConfiguration(BaseModel):

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = Field(
        default=(os.environ["EMBEDDING_MODEL"]),
        description="Name of the embedding model to use. Must be a valid embedding model name.",
    )

    retriever_provider: Annotated[
        Literal["elastic", "elastic-local"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = Field(
        default="elastic",
        description="The vector store provider to use for retrieval. Options are 'elastic', 'pinecone', or 'mongodb'.",
    )

    search_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the search function of the retriever.",
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "IndexConfiguration":
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        return cls(**{k: v for k, v in configurable.items() if k in cls.model_fields})


class Configuration(IndexConfiguration):
    response_system_prompt: str = Field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        description="The system prompt used for generating responses.",
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = Field(
        default=(os.environ["RESPONSE_MODEL"]),
        description="The language model used for generating responses. Should be in the form: provider/model-name.",
    )

    query_system_prompt: str = Field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        description="The system prompt used for processing and refining queries.",
    )

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = Field(
        default=(os.environ["QUERY_MODEL"]),
        description="The language model used for processing and refining queries. Should be in the form: provider/model-name.",
    )
    retrieve_decision_prompt: str = (
        "Decidi se è necessario cercare nei documenti per rispondere alla domanda dell'utente.\n"
        "Rispondi `true` se la domanda riguarda contenuti {SPECIFICA_ARGOMENTO}, ecc.\n"
        "Altrimenti rispondi `false`."
    )
    direct_answer_system_prompt: str = (
        "Ti chiami Anton e rispondi direttamente alla domanda dell'utente a patto che sia inerente a {SPECIFICA_ARGOMENTO}.\n"
    )
