import json
from typing import Dict

class PMLMessage:
    """Represents a PML (Prompt Markup Language) message sent by an agent."""

    def __init__(self, agent_name: str, query: str, result: str, relevance: float, agent_port: int):
        """
        Initialize the PML message.

        Args:
            agent_name (str): The name of the agent.
            query (str): The query the agent processed.
            result (str): The result (sentiment classification).
            relevance (float): The relevance score of the result.
            agent_port (int): The dynamically assigned port of the agent.
        """
        self.agent_name = agent_name
        self.query = query
        self.result = result
        self.relevance = relevance
        self.agent_port = agent_port

    def to_dict(self) -> Dict:
        """Convert the PML message to a dictionary for transmission."""
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "result": self.result,
            "relevance": self.relevance,
            "agent_port": self.agent_port
        }

    def to_json(self) -> str:
        """Convert the PML message to JSON format."""
        return json.dumps(self.to_dict())

    def __str__(self):
        return f"PMLMessage(agent={self.agent_name}, relevance={self.relevance}, port={self.agent_port})"
