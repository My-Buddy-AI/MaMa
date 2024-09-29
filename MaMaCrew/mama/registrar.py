import json
import os
import asyncio
from .network import receive_message, send_message
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

PML_STORAGE_FILE = 'pml_data.json'
QUERY_HISTORY_FILE = 'query_history.json'

class MAMARegistrar:
    """
    The MAMA Registrar Service that receives agent registrations via PML and dynamically evaluates them.
    Tracks the relevance and popularity of agents based on the received queries.
    """

    def __init__(self, port: int = 8089):
        """
        Initialize the MAMA registrar, load PML data and query history, and set up the sentence transformer model.
        
        Args:
            port (int): Port number on which the registrar will listen for messages.
        """
        self.port = port
        self.agent_registry = self.load_pml_data()  # Load PML data from storage
        self.query_history = self.load_query_history()  # Load query history from persistent storage
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence embedding model for semantic search
        print(f"MAMA Registrar initialized on port {self.port}.")

    async def listen_for_registration(self):
        """Asynchronously listen for incoming PML messages for agent registration."""
        print(f"Listening for agent registrations on port {self.port}...")
        while True:
            try:
                message = await receive_message(self.port)
                if message:
                    print(f"Received registration message: {message}")
                    await self.register_agent(message)
                else:
                    print("Received empty or invalid message.")
            except Exception as e:
                print(f"Error in receiving or processing message: {e}")
                import traceback
                traceback.print_exc()

    async def register_agent(self, pml_message: dict):
        """
        Asynchronously process a received PML message and register the agent.
        
        Args:
            pml_message (dict): The PML message containing agent details.
        """
        try:
            agent_name = pml_message['agent_name']
            relevance_score = pml_message['relevance']
            agent_address = pml_message['address']
            agent_port = pml_message['port']
            expertise_profile = pml_message.get('expertise_profile', {})
            prompt = pml_message.get('prompt', '')  # Agent prompt for evaluation

            # Register or update the agent's details in the registry
            self.agent_registry[agent_name] = {
                'relevance_score': relevance_score,
                'address': agent_address,
                'port': agent_port,
                'expertise_profile': expertise_profile,
                'prompt': prompt
            }

            # Save the updated registry to persistent storage
            self.save_pml_data()
            print(f"Registered agent '{agent_name}' at {agent_address}:{agent_port} with relevance score: {relevance_score}")
        except KeyError as e:
            print(f"Missing key in registration message: {e}")
        except Exception as e:
            print(f"Error registering agent: {e}")
            import traceback
            traceback.print_exc()

    async def evaluate_agents(self, query: str, sentiment: str):
        """
        Asynchronously evaluate all registered agents and return the most relevant agent based on sentiment and prompt matching.
        
        Args:
            query (str): The input query.
            sentiment (str): The sentiment type (e.g., 'positive', 'negative', 'sarcasm').

        Returns:
            tuple: The best agent's name, address, and port or None if no agent is found.
        """
        best_agent = None
        highest_similarity = -1

        try:
            # Convert query into embedding vector using the transformer model
            query_embedding = self.vectorize_text(query)

            # Evaluate agents based on sentiment and prompt similarity
            for agent_name, data in self.agent_registry.items():
                agent_prompt = data.get('prompt', '')
                agent_profile = data.get('expertise_profile', {})

                # Ensure the agent specializes in this sentiment
                if sentiment in agent_profile and agent_profile[sentiment] > 0:
                    # Compute similarity between agent's prompt and the query
                    prompt_embedding = self.vectorize_text(agent_prompt)
                    similarity = cosine_similarity([query_embedding], [prompt_embedding])[0][0]

                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_agent = (agent_name, data['address'], data['port'])

            if best_agent:
                print(f"Best agent selected for sentiment '{sentiment}' in query '{query}': {best_agent[0]} with similarity {highest_similarity}")
                # Store the successful query in history
                await self.store_query(query, best_agent[0])
                return best_agent
            else:
                print(f"No agents available for sentiment '{sentiment}' in query '{query}'")
                return None

        except Exception as e:
            print(f"Error evaluating agents: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def store_query(self, query: str, agent_name: str):
        """
        Store a successful query and the agent that responded in persistent query history.

        Args:
            query (str): The query that was handled.
            agent_name (str): The name of the agent that handled the query.
        """
        try:
            if agent_name not in self.query_history:
                self.query_history[agent_name] = []
            self.query_history[agent_name].append(query)

            # Save the query history to persistent storage
            self.save_query_history()
        except Exception as e:
            print(f"Error storing query: {e}")
            import traceback
            traceback.print_exc()

    def vectorize_text(self, text: str) -> np.ndarray:
        """
        Convert text into a vector embedding using the SentenceTransformer model.

        Args:
            text (str): The input text to vectorize.

        Returns:
            np.ndarray: The vector representation of the input text.
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            print(f"Error vectorizing text '{text}': {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((1, 384))  # Return a zero vector on error

    def save_pml_data(self):
        """Save the PML data to a file for permanent storage."""
        try:
            with open(PML_STORAGE_FILE, 'w') as file:
                json.dump(self.agent_registry, file)
            print(f"PML data saved to {PML_STORAGE_FILE}.")
        except Exception as e:
            print(f"Error saving PML data: {e}")
            import traceback
            traceback.print_exc()

    def load_pml_data(self):
        """Load the PML data from a file (if exists) or initialize an empty registry."""
        try:
            if os.path.exists(PML_STORAGE_FILE):
                with open(PML_STORAGE_FILE, 'r') as file:
                    return json.load(file)
            return {}  # Return an empty dictionary if no file exists
        except Exception as e:
            print(f"Error loading PML data: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def save_query_history(self):
        """Save the query history to a file for persistent storage."""
        try:
            with open(QUERY_HISTORY_FILE, 'w') as file:
                json.dump(self.query_history, file)
            print(f"Query history saved to {QUERY_HISTORY_FILE}.")
        except Exception as e:
            print(f"Error saving query history: {e}")
            import traceback
            traceback.print_exc()

    def load_query_history(self):
        """Load the query history from a file or initialize an empty history."""
        try:
            if os.path.exists(QUERY_HISTORY_FILE):
                with open(QUERY_HISTORY_FILE, 'r') as file:
                    return json.load(file)
            return {}  # Return an empty dictionary if no file exists
        except Exception as e:
            print(f"Error loading query history: {e}")
            import traceback
            traceback.print_exc()
            return {}
