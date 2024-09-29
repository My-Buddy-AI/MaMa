import json
import os
from .network import receive_message

PML_STORAGE_FILE = 'pml_data.json'

class MAMARegistrar:
    """
    The MAMA Registrar Service that receives agent registrations via PML and dynamically evaluates them.
    Tracks the relevance and popularity of agents based on the received queries.
    """

    def __init__(self, port: int = 8089):
        self.port = port
        self.agent_registry = self.load_pml_data()  # Load PML data from storage
        print(f"MAMA Registrar initialized on port {self.port}.")

    def listen_for_registration(self):
        """Listen for incoming PML messages for agent registration."""
        print(f"Registrar listening for PML messages on port {self.port}...")
        while True:
            pml_message = receive_message(self.port)
            self.register_agent(pml_message)

    def register_agent(self, pml_message: dict):
        """Process a received PML message and register the agent."""
        agent_name = pml_message['agent_name']
        relevance_score = pml_message['relevance']
        agent_address = pml_message['address']
        agent_port = pml_message['port']
        expertise_profile = pml_message.get('expertise_profile', {})

        # Register or update the agent's details in the registry
        self.agent_registry[agent_name] = {
            'relevance_score': relevance_score,
            'address': agent_address,
            'port': agent_port,
            'expertise_profile': expertise_profile
        }

        # Save the updated registry to persistent storage
        self.save_pml_data()
        print(f"Registered agent '{agent_name}' at {agent_address}:{agent_port} with relevance score: {relevance_score}")

    def evaluate_agents(self, query: str, sentiment: str):
        """
        Evaluate all registered agents and return the most relevant agent based on sentiment.

        Args:
            query (str): The input query.
            sentiment (str): The sentiment type (e.g., 'positive', 'negative', 'sarcasm').

        Returns:
            tuple: The best agent's name, address, and port or None if no agent is found.
        """
        best_agent = None
        highest_relevance = -1

        # Evaluate agents based on the sentiment type
        for agent_name, data in self.agent_registry.items():
            agent_profile = data.get('expertise_profile', {})
            
            # Check if the agent's expertise matches the sentiment
            if sentiment in agent_profile and agent_profile[sentiment] > 0:
                relevance_score = data['relevance_score']
                if relevance_score > highest_relevance:
                    highest_relevance = relevance_score
                    best_agent = (agent_name, data['address'], data['port'])

        if best_agent:
            print(f"Best agent selected for sentiment '{sentiment}' in query '{query}': {best_agent[0]} with relevance {highest_relevance}")
            return best_agent
        else:
            print(f"No agents available for sentiment '{sentiment}' in query '{query}'")
            return None

    def remove_agent(self, agent_name: str):
        """Remove an agent from the registry."""
        if agent_name in self.agent_registry:
            del self.agent_registry[agent_name]
            self.save_pml_data()
            print(f"Agent '{agent_name}' has been removed from the registry.")
        else:
            print(f"Agent '{agent_name}' not found in the registry.")

    def update_agent(self, agent_name: str, pml_message: dict):
        """Update an agent's details in the registry."""
        if agent_name in self.agent_registry:
            self.agent_registry[agent_name].update(pml_message)
            self.save_pml_data()
            print(f"Agent '{agent_name}' has been updated.")
        else:
            print(f"Agent '{agent_name}' not found. Registering as a new agent.")
            self.register_agent(pml_message)

    def save_pml_data(self):
        """Save the PML data to a file for permanent storage."""
        with open(PML_STORAGE_FILE, 'w') as file:
            json.dump(self.agent_registry, file)

    def load_pml_data(self):
        """Load the PML data from a file (if exists) or initialize an empty registry."""
        if os.path.exists(PML_STORAGE_FILE):
            with open(PML_STORAGE_FILE, 'r') as file:
                return json.load(file)
        return {}  # Return an empty dictionary if no file exists
