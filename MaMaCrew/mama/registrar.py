import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from .network import receive_message

PML_STORAGE_FILE = 'pml_data.json'
QUERY_HISTORY_FILE = 'query_history.json'

class MAMARegistrar:
    """
    The MAMA Registrar Service that receives agent registrations via PML and dynamically evaluates them.
    Tracks the relevance and popularity of agents based on the received queries and persistently stores the query history.
    """

    def __init__(self, port: int = 8089):
        
        self.port = port
        self.agent_registry = self.load_pml_data()  # Load PML data from storage
        self.query_history = self.load_query_history()  # Load query history from persistent storage
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained model for sentence embeddings
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
        self.agent_registry = self.load_pml_data()  
        print(f"Registered agent '{agent_name}' at {agent_address}:{agent_port} with relevance score: {relevance_score}")

    def is_query_within_scope(self, query: str, agent_data: dict) -> bool:
        """
        Check if the provided query matches the scope of the agent based on its specialty, profile, or prompt.

        Args:
            query (str): The query to check.
            agent_data (dict): The data of the agent (loaded from YAML) which includes its specialty, profile, etc.

        Returns:
            bool: True if the query matches the agent's scope, False otherwise.
        """
        specialty = agent_data.get('specialty', '').lower()
        profile = agent_data.get('profile', {})

        # Check for relevant terms or phrases in the query to see if they align with the agent's specialty
        # For the mischievousness agent, we would look for rebellious, playful, or gross-out humor
        if "mischievous" in specialty or "rebellion" in specialty or "humor" in specialty:
            if any(term in query.lower() for term in ["hide", "secrecy", "secretions", "parental", "mischief"]):
                return True

        # If there are specific characteristics in the profile that we want to match
        sarcasm_level = profile.get('sarcasm', 0)
        if sarcasm_level > 0.5 and "sarcasm" in query.lower():
            return True

        # Add other specific checks if needed
        return False

    def training(self, query: str):
        """
        Train all agents with the provided query. Each agent's relevance score (similarity)
        is calculated based on the query. Agents with a relevance score greater than 70%
        will store the query in their history.

        Args:
            query (str): The query to train on.
        
        Returns:
            list: A list of agents who had a similarity score above 70%.
        """
        relevant_agents = []
        similarity_threshold = 0.7  # 70% similarity threshold

        # Vectorize the query (convert to embedding for similarity calculation)
        query_embedding = self.vectorize_text(query)

        # Load all agents' configuration data from their respective YAML files
        self.agent_registry = self.load_pml_data()

        for agent_name, data in self.agent_registry.items():
            # Check if the query falls within the agent's scope
            if not self.is_query_within_scope(query, data):
                print(f"Query '{query}' is not within the scope of agent '{agent_name}'")
                continue  # Skip to the next agent if the query doesn't match the scope

            # Retrieve agent-specific configurations
            agent_prompt = data.get('prompt', '')
            
            # Vectorize the agent's prompt for similarity calculation
            agent_prompt_embedding = self.vectorize_text(agent_prompt)
            
            # Calculate the similarity between the query and the agent's prompt
            similarity_score = cosine_similarity([query_embedding], [agent_prompt_embedding])[0][0]
            
            # Check if the similarity score is greater than the threshold
            if similarity_score >= similarity_threshold:
                print(f"Agent '{agent_name}' has a similarity score of {similarity_score:.2f} for query: '{query}'")
                
                # Add the query to this agent's history
                self.add_query_to_history(agent_name, query)
                print(f"Query '{query}' added to the history of agent '{agent_name}'")
                relevant_agents.append(agent_name)

        if relevant_agents:
            print(f"Agents trained with query '{query}': {', '.join(relevant_agents)}")
            return relevant_agents
        else:
            print(f"No agents found with sufficient similarity for query '{query}'")
            return None

    def evaluate_agents(self, query: str):
        """
        Evaluate all registered agents and return the most relevant agent based on prompt similarity and query history.
        If no agent matches the query with sufficient similarity, return None.

        Args:
            query (str): The input query.

        Returns:
            tuple or None: The best agent's name, address, and port or None if no suitable agent is found.
        """
        best_agent = None
        highest_similarity = -1
        similarity_threshold = 0.5  # Set a minimum similarity threshold to consider an agent

        query_embedding = self.vectorize_text(query)

        # Evaluate agents based on the prompt similarity
        self.agent_registry = self.load_pml_data()  
        for agent_name, data in self.agent_registry.items():
            agent_prompt = data.get('prompt', '')

            # Compute similarity between the query and the agent's prompt
            agent_prompt_embedding = self.vectorize_text(agent_prompt)
            similarity_score = cosine_similarity([query_embedding], [agent_prompt_embedding])[0][0]

            # If the agent has answered a similar query before, increase the score
            if self.has_agent_answered_similar_query(agent_name, query):
                similarity_score += 0.1  # Bonus for handling similar queries

            # Only consider agents with a similarity score above the threshold
            if similarity_score >= highest_similarity and similarity_score >= similarity_threshold:
                highest_similarity = similarity_score
                best_agent = (agent_name, data['address'], data['port'])

        if best_agent:
            print(f"Best agent selected for query '{query}': {best_agent[0]} with similarity {highest_similarity}")
            # Save this query to the agent's history and persist it
            self.add_query_to_history(best_agent[0], query)
            return best_agent
        else:
            print(f"No agents available with sufficient similarity for query '{query}'")
            return None
    
    def has_agent_answered_similar_query(self, agent_name: str, query: str) -> bool:
        """
        Check if an agent has answered a similar query in the past.

        Args:
            agent_name (str): The agent's name.
            query (str): The new query to be checked.

        Returns:
            bool: True if the agent has handled a similar query, False otherwise.
        """
        if agent_name not in self.query_history:
            return False

        # Use a similarity check for historical queries (cosine similarity)
        query_embedding = self.vectorize_text(query)
        for past_query in self.query_history[agent_name]:
            past_query_embedding = self.vectorize_text(past_query)
            similarity_score = cosine_similarity([query_embedding], [past_query_embedding])[0][0]
            if similarity_score > 0.8:  # Similar query threshold
                return True

        return False

    def add_query_to_history(self, agent_name: str, query: str):
        """Add a query to the agent's query history and persist it."""
        if agent_name not in self.query_history:
            self.query_history[agent_name] = []
        self.query_history[agent_name].append(query)
        self.save_query_history()  # Persist the updated history

    def vectorize_text(self, text: str):
        """
        Convert the text into a vector using a sentence transformer (pre-trained model).

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: The vectorized representation of the input text.
        """
        return self.model.encode(text)

    def find_next_best_agent(self, query: str, sentiment: str):
        """
        Find the next best agent if the current best agent is not suitable.

        Args:
            query (str): The input query.
            sentiment (str): The sentiment extracted from the query.

        Returns:
            tuple: The best agent's name, address, and port or None if no suitable agent is found.
        """
        # Iterate over agents again to find the next best option
        next_best_agent = None
        highest_similarity = -1

        query_embedding = self.vectorize_text(query)

        for agent_name, data in self.agent_registry.items():
            agent_profile = data.get('expertise_profile', {})
            agent_prompt = data.get('prompt', '')

            if sentiment in agent_profile and agent_profile[sentiment] > 0:
                agent_prompt_embedding = self.vectorize_text(agent_prompt)
                similarity_score = cosine_similarity([query_embedding], [agent_prompt_embedding])[0][0]

                if similarity_score > highest_similarity:
                    highest_similarity = similarity_score
                    next_best_agent = (agent_name, data['address'], data['port'])

        if next_best_agent:
            print(f"Next best agent selected for sentiment '{sentiment}' in query '{query}': {next_best_agent[0]} with similarity {highest_similarity}")
            return next_best_agent
        else:
            print(f"No more agents available for sentiment '{sentiment}' in query '{query}'")
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

    def save_query_history(self):
        """Save the query history to a file for persistent storage."""
        with open(QUERY_HISTORY_FILE, 'w') as file:
            json.dump(self.query_history, file)

    def load_query_history(self):
        """Load the query history from a file or initialize an empty history."""
        if os.path.exists(QUERY_HISTORY_FILE):
            with open(QUERY_HISTORY_FILE, 'r') as file:
                return json.load(file)
        return {}  # Return an empty dictionary if no file exists