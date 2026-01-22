"""Language model integration service."""

from typing import List, Dict


class LLMService:
    """Handle LLM interactions for question answering."""

    def __init__(self, model_name: str = "llama3", base_url: str = ""):
        """
        Initialize the LLM service.

        Args:
            model_name: Name of the LLM model
            base_url: Base URL for the LLM server (e.g., Ollama)
        """
        self.model_name = model_name
        self.base_url = base_url or "http://localhost:11434"
        # TODO: Initialize LLM client

    def generate_response(  # type: ignore
        self, question: str, context: List[Dict[str, str]]
    ) -> str:
        """
        Generate a response to a question given context.

        Args:
            question: User's question
            context: Retrieved document chunks

        Returns:
            Generated response
        """
        # TODO: Implement response generation
        pass

    def format_prompt(self, question: str, context: List[Dict[str, str]]) -> str:  # type: ignore
        """
        Format the prompt for the LLM.

        Args:
            question: User's question
            context: Retrieved document chunks

        Returns:
            Formatted prompt string
        """
        # TODO: Implement prompt formatting
        pass
