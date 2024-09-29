import random
import socket
import time  # Import time to replace asyncio.sleep for synchronous waiting
import yaml
from typing import Dict
from .network import send_message
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .pml import PMLMessage


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

        self.register_with_mama()  # Register synchronously with retries

    def load_config(self, config_path: str):
        """Load the agent's configuration from the YAML file."""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        self.name = config['name']
        self.profile = config['profile']
        self.specialty = config.get('specialty', 'general')  # New addition for specialty

    def assign_dynamic_port(self):
        """Assign an available port dynamically."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # Bind to an available port
        port = s.getsockname()[1]  # Get the dynamically assigned port
        s.close()
        return port

    def register_with_mama(self):
        """Register the agent with the MAMA registrar using PML. Retry on failure with exponential backoff."""
        max_retries = 10  # Maximum number of retries
        retries = 0
        base_wait_time = 0.1  # Initial wait time of 100ms
        success = False

        while not success and retries < max_retries:
            try:
                print(f"Attempt {retries + 1}: Registering {self.name} with MAMA registrar at port {self.receive_port}...")

                # Get the agent's IP address
                hostname = socket.gethostname()
                agent_address = socket.gethostbyname(hostname)

                pml_data = {
                    'agent_name': self.name,
                    'expertise_profile': self.profile,
                    'specialty': self.specialty,  # Register the specialty of the agent
                    'relevance': self.relevance_score,
                    'address': agent_address,
                    'port': self.receive_port
                }

                # Send registration request via PML to Registrar
                send_message(self.pml_port, pml_data)  # Synchronous message sending
                success = True
                print(f"Agent '{self.name}' successfully registered with MAMA.")

            except Exception as e:
                retries += 1
                print(f"Attempt {retries} failed: Could not register {self.name} on port {self.receive_port}. Error: {e}")

                # Exponential backoff: wait time doubles after each failed attempt
                wait_time = base_wait_time * (2 ** retries)
                if retries < max_retries:
                    print(f"Retrying in {wait_time:.2f} seconds...")

                    # Assign new dynamic ports to try again
                    self.receive_port = self.assign_dynamic_port()
                    self.reply_port = self.assign_dynamic_port()
                    print(f"Retrying registration with new ports: receive={self.receive_port}, reply={self.reply_port}")

                    time.sleep(wait_time)  # Synchronous wait before retrying
                else:
                    print(f"Exceeded max retries. Could not register agent '{self.name}' after {retries} attempts.")
                    return  # Final failure

        if success:
            print(f"Registration of '{self.name}' with MAMA was successful after {retries + 1} attempts.")
        else:
            print(f"Registration failed after {max_retries} attempts. Please check the network or registrar service.")

    def train_agent(self, query: str, markup: Dict[str, float]):
        """Train the agent based on the input query and markup."""
        print(f"Training {self.name} with query: {query}")
        if "positive" in query:
            self.relevance_score += 0.1  # Increase the relevance for positive queries
        elif "negative" in query:
            self.relevance_score += 0.1  # Increase the relevance for negative queries
        elif "sarcasm" in query:
            self.relevance_score += 0.1  # Increase the relevance for sarcasm
        elif "neutral" in query:
            self.relevance_score += 0.1  # Increase the relevance for neutral queries

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
            "complexity": 0.5 if "complex" in query else 0.2,
            "neutral": 0.9 if "neutral" in query else 0.2
        }
        return markup

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
