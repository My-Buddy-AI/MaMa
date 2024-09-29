import json

class PMLMessage:
    """
    Represents a PML (Prompt Markup Language) message that agents use to communicate with the MAMA registrar.
    The message includes details about the agent, the query, the result, and the relevance score.
    """

    def __init__(self, agent_name: str, query: str, result: str, relevance: float, agent_port: int):
        """
        Initialize the PML message with agent details, query, result, relevance score, and the agent's port.

        Args:
            agent_name (str): The name of the agent.
            query (str): The query being processed.
            result (str): The result generated by the agent.
            relevance (float): The relevance score of the agent's response to the query.
            agent_port (int): The port where the agent is listening for responses.
        """
        self.agent_name = agent_name
        self.query = query
        self.result = result
        self.relevance = relevance
        self.agent_port = agent_port

    def to_dict(self) -> dict:
        """
        Convert the PML message to a dictionary format for transmission.

        Returns:
            dict: The PML message serialized as a dictionary.
        """
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "result": self.result,
            "relevance": self.relevance,
            "agent_port": self.agent_port
        }

    def to_json(self) -> str:
        """
        Convert the PML message to a JSON string for transmission.

        Returns:
            str: The PML message serialized as a JSON string.
        """
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(json_string: str):
        """
        Create a PMLMessage object from a JSON string.

        Args:
            json_string (str): The JSON string to be parsed into a PMLMessage.

        Returns:
            PMLMessage: The PML message object parsed from the JSON string.
        """
        data = json.loads(json_string)
        return PMLMessage(
            agent_name=data['agent_name'],
            query=data['query'],
            result=data['result'],
            relevance=data['relevance'],
            agent_port=data['agent_port']
        )
