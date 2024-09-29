import random
import socket
import yaml
from typing import Dict
from .pml import PMLMessage
from .network import send_message
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CrewAIAgent:
    """
    CrewAI agent that dynamically registers with MAMA using PML. It updates its relevance score
    based on reinforcement learning feedback after each query. It also handles agent similarity
    when no direct match is found.
    """

    def __init__(self, config_path: str, training_mode=False):
        """
        Initialize the CrewAI agent from a YAML configuration file.

        Args:
            config_path (str): The path to the agent's YAML configuration file.
            training_mode (bool): Whether the agent is in training mode.
        """
        self.training_mode = training_mode
        self.config_path = config_path
        self.load_config(config_path)
        self.popularity = 0
        self.relevance_score = 0.0  # Initial relevance score for RL evaluation

        # Dynamically assign ports for MAMA communication
        self.receive_port = self.assign_dynamic_port()
        self.reply_port = self.assign_dynamic_port()
        self.pml_port = 8089  # Registrar is hosted at port 8089

        print(f"Agent '{self.name}' initialized with ports: receive={self.receive_port}, reply={self.reply_port}, PML={self.pml_port}")

        # Register agent with MAMA registrar using PML
        self.register_with_mama()

    def load_config(self, config_path: str):
        """Load the agent's configuration from the YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.name = config['name']
        self.profile = config['profile']

    def assign_dynamic_port(self):
        """Assign an available port dynamically."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # Bind to an available port
        port = s.getsockname()[1]  # Get the dynamically assigned port
        s.close()
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

        # If in training mode, update the agent's knowledge
        if self.training_mode:
            self.train_agent(query, markup)

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

    def train_agent(self, query: str, markup: Dict[str, float]):
        """Train the agent based on the input query and markup."""
        print(f"Training {self.name} with query: {query}")
        if "positive" in query:
            self.relevance_score += 0.1  # Increase the relevance for positive queries
        elif "negative" in query:
            self.relevance_score += 0.1  # Increase the relevance for negative queries
        elif "sarcasm" in query:
            self.relevance_score += 0.1  # Increase the relevance for sarcasm

    def process_query(self, query: str, markup: Dict[str, float]) -> str:
        """Process the query based on the content and return a response."""
        # Simulate query handling based on profile (simplified for demo purposes)
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

    def update_relevance_with_reinforcement(self, relevance: float):
        """Use reinforcement learning to adjust the relevance score based on feedback."""
        feedback = random.choice([1, -1])  # Simulate positive or negative feedback
        self.relevance_score += feedback * relevance
        self.relevance_score = max(0, self.relevance_score)  # Keep score non-negative
        print(f"Updated relevance score for {self.name}: {self.relevance_score}")

    def extract_markup(self, query: str) -> Dict[str, float]:
        """Extract markup prompts from the input query."""
        # Simplified sentiment tagging logic
        markup = {
            "positive": 0.8 if "good" in query or "happy" in query else 0.2,
            "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1,
            "complexity": 0.5 if "complex" in query else 0.2
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

    @staticmethod
    def calculate_similarity(agent_profile: Dict[str, float], query_markup: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between agent's profile and query markup.
        
        Args:
            agent_profile: The agent's sentiment profile (e.g., {"positive": 1.0, "negative": 0.2})
            query_markup: The query markup extracted from the input query.
        
        Returns:
            float: The cosine similarity score.
        """
        profile_vector = np.array(list(agent_profile.values())).reshape(1, -1)
        markup_vector = np.array(list(query_markup.values())).reshape(1, -1)
        similarity_score = cosine_similarity(profile_vector, markup_vector)[0][0]
        return similarity_score
