import json

class PMLMessage:
    """
    PMLMessage represents the Prompt Markup Language message format that agents use to register with MAMA.
    It encapsulates the necessary information such as the agent's name, expertise profile, relevance score, and address.
    """

    def __init__(self, agent_name: str, query: str, result: str, relevance: float, agent_port: int, address: str = None):
        """
        Initialize a PMLMessage instance.

        Args:
            agent_name (str): The name of the agent.
            query (str): The query the agent responded to.
            result (str): The result returned by the agent.
            relevance (float): The relevance score of the agent for the query.
            agent_port (int): The port on which the agent is running.
            address (str, optional): The address of the agent. Defaults to None.
        """
        self.agent_name = agent_name
        self.query = query
        self.result = result
        self.relevance = relevance
        self.agent_port = agent_port
        self.address = address

    def to_dict(self):
        """
        Convert the PMLMessage instance to a dictionary.

        Returns:
            dict: A dictionary representation of the PMLMessage.
        """
        return {
            "agent_name": self.agent_name,
            "query": self.query,
            "result": self.result,
            "relevance": self.relevance,
            "agent_port": self.agent_port,
            "address": self.address
        }

    @classmethod
    def from_dict(cls, data: dict):
        """
        Create a PMLMessage instance from a dictionary.

        Args:
            data (dict): A dictionary containing the PMLMessage data.

        Returns:
            PMLMessage: An instance of the PMLMessage class.
        """
        return cls(
            agent_name=data['agent_name'],
            query=data['query'],
            result=data['result'],
            relevance=data['relevance'],
            agent_port=data['agent_port'],
            address=data.get('address')
        )

    def to_json(self) -> str:
        """
        Convert the PMLMessage instance to a JSON string.

        Returns:
            str: A JSON string representation of the PMLMessage.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str):
        """
        Create a PMLMessage instance from a JSON string.

        Args:
            json_str (str): A JSON string containing the PMLMessage data.

        Returns:
            PMLMessage: An instance of the PMLMessage class.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self):
        """
        Represent the PMLMessage as a string for debugging purposes.

        Returns:
            str: A string representation of the PMLMessage instance.
        """
        return f"PMLMessage(agent_name={self.agent_name}, query={self.query}, result={self.result}, relevance={self.relevance}, agent_port={self.agent_port}, address={self.address})"
