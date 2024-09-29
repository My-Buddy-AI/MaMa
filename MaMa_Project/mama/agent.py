import random
import socket
from typing import Dict
from .pml import PMLMessage
from .network import send_message

class CrewAIAgent:
    """
    Base class for CrewAI agents with dynamic port assignment, prompt-based profiles, and PML communication.
    This agent can be transformed into a dynamically enabled MAMA agent.
    """

    def __init__(self, agent_prompt: str, profile: Dict[str, float]):
        """
        Initialize a CrewAI agent dynamically based on the given prompt.

        Args:
            agent_prompt (str): The role or prompt that defines the agent's expertise (e.g., 'Positive Classifier').
            profile (Dict[str, float]): Profile describing the agent's expertise on different sentiment types.
        """
        self.name = f"CrewAI Agent - {agent_prompt}"  # The agent's dynamic name based on its prompt
        self.profile = profile  # Expertise profile (e.g., {"positive": 0.8, "negative": 0.2})
        self.popularity = 0

        # Dynamically assign ports for MAMA communication
        self.receive_port = self.assign_dynamic_port()
        self.reply_port = self.assign_dynamic_port()
        self.pml_port = self.assign_dynamic_port()

        print(f"Agent '{self.name}' initialized with ports: receive={self.receive_port}, reply={self.reply_port}, PML={self.pml_port}")

    def assign_dynamic_port(self):
        """Assign an available port dynamically."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # Bind to an available port
        port = s.getsockname()[1]  # Get the dynamically assigned port
        s.close()  # Release the socket
        return port

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
        # Basic example: positive, negative, or neutral based on query content
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
        pml_message = PMLMessage(agent_name=self.name, query=query, result=result, relevance=relevance, agent_port=self.reply_port)
        print(f"Agent {self.name} sending PML: {pml_message}")
        send_message(self.pml_port, pml_message.to_dict())

    def transform_to_mama_agent(self):
        """
        Transform the CrewAI agent into a dynamic MAMA agent, enabling it to be part of the MAMA framework.
        """
        print(f"Transforming {self.name} into a MAMA-enabled agent.")
