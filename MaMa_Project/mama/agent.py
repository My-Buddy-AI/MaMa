import random
from typing import Dict
from .pml import PMLMessage
from .network import send_message


class AIAgent:
    """Base class for AI Agents with profiles, ports, and relevance calculation."""

    def __init__(self, name: str, receive_port: int, reply_port: int, pml_port: int, profile: Dict[str, float]):
        """
        Initialize the agent with ports and a profile.

        Args:
            name (str): The name of the agent.
            receive_port (int): Port to receive requests.
            reply_port (int): Port to send replies.
            pml_port (int): Port to send PML messages to the registrar.
            profile (Dict[str, float]): Profile describing expertise on different sentiment types.
        """
        self.name = name
        self.receive_port = receive_port  # Port to receive requests
        self.reply_port = reply_port      # Port to send replies
        self.pml_port = pml_port          # Port to send PML messages
        self.profile = profile            # Agent's expertise profile (e.g., {"positive": 0.8, "negative": 0.2})
        self.popularity = 0               # Popularity of the agent, evaluated by the Registrar

    def receive_request(self, query: str):
        """Receive a query, process it, and send the PML message."""
        print(f"Agent {self.name} received query: {query}")
        
        # Extract markup prompts from the query
        markup = self.extract_markup(query)

        # Calculate relevance score based on the agent's profile and the markup
        relevance = self.evaluate(query, markup)
        
        # Generate a result based on the query and markup
        result = self.process_query(query, markup)

        # Send the result back to the client
        self.send_reply(result)

        # Send the PML message with the relevance score to the Registrar
        self.send_pml(query, result, relevance)

    def process_query(self, query: str, markup: Dict[str, float]) -> str:
        """Process the query based on the content and return a response."""
        # Simple result determination based on query content
        if "good" in query:
            return "positive"
        elif "bad" in query:
            return "negative"
        return "neutral"

    def evaluate(self, query: str, markup: Dict[str, float]) -> float:
        """Calculate relevance score based on the agent's profile and the query markup."""
        relevance = 0.0
        for tag, weight in markup.items():
            if tag in self.profile:
                relevance += self.profile[tag] * weight
        return relevance

    def extract_markup(self, query: str) -> Dict[str, float]:
        """Extract markup prompts from the input query."""
        markup = {
            "positive": 0.8 if "good" in query or "happy" in query else 0.2,
            "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1
        }
        return markup

    def send_reply(self, result: str):
        """Send the reply to the client."""
        print(f"Agent {self.name} sending reply: {result}")
        send_message(self.reply_port, {"agent": self.name, "result": result})

    def send_pml(self, query: str, result: str, relevance: float):
        """Send a PML (Prompt Markup Language) message to the Registrar."""
        pml_message = PMLMessage(agent_name=self.name, query=query, result=result, relevance=relevance)
        print(f"Agent {self.name} sending PML: {pml_message}")
        send_message(self.pml_port, pml_message.to_dict())


class PositiveClassifier(AIAgent):
    def __init__(self, receive_port: int, reply_port: int, pml_port: int):
        profile = {"positive": 1.0, "negative": 0.2, "sarcasm": 0.1}
        super().__init__("Positive Classifier", receive_port, reply_port, pml_port, profile)


class NegativeClassifier(AIAgent):
    def __init__(self, receive_port: int, reply_port: int, pml_port: int):
        profile = {"positive": 0.2, "negative": 1.0, "sarcasm": 0.1}
        super().__init__("Negative Classifier", receive_port, reply_port, pml_port, profile)
