import random
import socket
from typing import Dict
from .pml import PMLMessage
from .network import send_message


class CrewAIAgent:
    """
    Base class for CrewAI agents that dynamically register with MAMA using PML.
    It also supports reinforcement learning to evaluate prompt marks dynamically.
    """

    def __init__(self, agent_prompt: str, profile: Dict[str, float]):
        """
        Initialize a CrewAI agent dynamically based on the given prompt.

        Args:
            agent_prompt (str): The role or prompt that defines the agent's expertise (e.g., 'Positive Classifier').
            profile (Dict[str, float]): Profile describing the agent's expertise on different sentiment types.
        """
        self.name = f"CrewAI Agent - {agent_prompt}"
        self.profile = profile  # Expertise profile (e.g., {"positive": 0.8, "negative": 0.2})
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

    # The rest of the class remains the same
