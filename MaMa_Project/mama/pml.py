import json
from typing import Dict

class PMLMessage:
    """Represents a PML (Prompt Markup Language) message sent by an agent."""

    def __init__(self, agent_name: str, query: str, result: str, relevance: float):
        self.agent_name = agent_name
        self.query = query
        self.result = result
        self.relevance = relevance

    def to_dict(self) -> Dict:
        """Convert the PML message to a dictionary for transmission."""
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "result": self.result,
            "relevance": self.relevance
        }

    def to_json(self) -> str:
        """Convert the PML message to JSON format."""
        return json.dumps(self.to_dict())

    def __str__(self):
        return f"PMLMessage(agent={self.agent_name}, relevance={self.relevance})"
