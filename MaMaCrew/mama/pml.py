import json

class PMLMessage:
    """
    Represents a PML (Prompt Markup Language) message sent by an agent to the MAMA Registrar.
    """

    def __init__(self, agent_name: str, query: str, result: str, relevance: float, agent_port: int):
        self.agent_name = agent_name
        self.query = query
        self.result = result
        self.relevance = relevance
        self.agent_port = agent_port

    def to_dict(self):
        """Convert the PML message to a dictionary for transmission."""
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "result": self.result,
            "relevance": self.relevance,
            "port": self.agent_port
        }

    def to_json(self):
        """Convert the PML message to a JSON string."""
        return json.dumps(self.to_dict())

    def __str__(self):
        return f"PMLMessage(agent={self.agent_name}, relevance={self.relevance}, port={self.agent_port})"