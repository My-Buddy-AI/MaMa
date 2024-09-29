import random
import socket
from typing import Dict
from .pml import PMLMessage
from .network import send_message


class MishmashStyleAgent:
    """
    An agent specialized in handling complex, mixed, or conflicting styles and genres.
    This agent can detect ambiguity, sarcasm, and complex tonal shifts within a single text.
    It registers dynamically with the MAMA registrar and uses reinforcement learning to improve over time.
    """

    def __init__(self, profile: Dict[str, float]):
        """
        Initialize the MishmashStyleAgent dynamically based on its profile.

        Args:
            profile (Dict[str, float]): Profile describing the agent's ability to handle conflicting styles, sentiments, etc.
        """
        self.name = "MishmashStyleAgent"
        self.profile = profile  # Expertise profile (e.g., {"positive": 0.3, "negative": 0.3, "sarcasm": 0.7, "complexity": 1.0})
        self.popularity = 0
        self.relevance_score = 0.0  # Initial relevance score for RL evaluation

        # Dynamically assign ports for MAMA communication
        self.receive_port = self.assign_dynamic_port()
        self.reply_port = self.assign_dynamic_port()
        self.pml_port = 8089  # Registrar is hosted at port 8089

        print(f"Agent '{self.name}' initialized with ports: receive={self.receive_port}, reply={self.reply_port}, PML={self.pml_port}")

        # Register agent with MAMA registrar using PML
        self.register_with_mama()

    def assign_dynamic_port(self):
        """Assign an available port dynamically."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # Bind to an available port
        port = s.getsockname()[1]  # Get the dynamically assigned port
        s.close()  # Release the socket
        return port

    def register_with_mama(self):
        """Register the agent with the MAMA registrar using PML."""
        print(f"Registering {self.name} with MAMA registrar.")

        # Get the agent's IP address
        hostname = socket.gethostname()
        agent_address = socket.gethostbyname(hostname)

        pml_data = {
            'agent_name': self.name,
            'expertise_profile': self.profile,
            'relevance': self.relevance_score,
            'address': agent_address,
            'port': self.receive_port
        }

        # Send registration request via PML to Registrar
        send_message(self.pml_port, pml_data)

    def receive_request(self, query: str):
        """Receive a query, process it, and send the PML message."""
        print(f"Agent {self.name} received query: {query}")

        # Extract markup prompts from the query
        markup = self.extract_markup(query)

        # Calculate relevance score based on the agent's profile and the markup
        relevance = self.evaluate(query, markup)

        # Update relevance score using reinforcement learning (e.g., positive feedback)
        self.update_relevance_with_reinforcement(relevance)

        # Generate a result based on the query and markup
        result = self.process_query(query, markup)

        # Send the result back to the client
        self.send_reply(result)

        # Send the PML message with the relevance score to the Registrar
        self.send_pml(query, result, relevance)

    def process_query(self, query: str, markup: Dict[str, float]) -> str:
        """Process the query based on the content and return a response."""
        # This agent can handle mixed styles, ambiguity, and sarcasm
        if "not" in query and "bad" in query:
            return "sarcasm"
        elif "good" in query and "bad" in query:
            return "ambiguous"
        elif "complex" in query or "mishmash" in query:
            return "complex"
        return "neutral"

    def evaluate(self, query: str, markup: Dict[str, float]) -> float:
        """Calculate relevance score based on the agent's profile and the query markup."""
        relevance = 0.0
        for tag, weight in markup.items():
            if tag in self.profile:
                relevance += self.profile[tag] * weight
        return relevance

    def update_relevance_with_reinforcement(self, relevance: float):
        """Use reinforcement learning to adjust the relevance score based on feedback."""
        feedback = random.choice([1, -1])  # Simulate positive or negative feedback
        self.relevance_score += feedback * relevance
        self.relevance_score = max(0, self.relevance_score)  # Keep score non-negative
        print(f"Updated relevance score for {self.name}: {self.relevance_score}")

    def extract_markup(self, query: str) -> Dict[str, float]:
        """Extract markup prompts from the input query."""
        markup = {
            "positive": 0.5 if "good" in query else 0.2,
            "negative": 0.5 if "bad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1,
            "complexity": 1.0 if "complex" in query or "mishmash" in query else 0.1
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
