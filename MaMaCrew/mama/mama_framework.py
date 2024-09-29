from .registrar import MAMARegistrar

class MAMAFramework:
    """
    MAMAFramework manages dynamically added CrewAI agents and processes queries by interacting with MAMA Registrar.
    It discovers the most relevant agent for each query and allows dynamic addition, updating, and removal of agents.
    """

    def __init__(self, registrar: MAMARegistrar):
        """Initialize the MAMA framework with a reference to the MAMA Registrar."""
        self.registrar = registrar
        self.agents = {}  # Dictionary to store agents by name

    def add_agent(self, crewai_agent):
        """
        Add a new CrewAI agent to the framework.

        Args:
            crewai_agent: The CrewAI agent to be added to the framework.
        """
        if crewai_agent.name in self.agents:
            print(f"Agent '{crewai_agent.name}' already exists in the MAMA framework.")
        else:
            self.agents[crewai_agent.name] = crewai_agent
            print(f"Agent '{crewai_agent.name}' added to the MAMA framework.")

    def update_agent(self, crewai_agent):
        """
        Update the profile or properties of an existing agent in the framework.

        Args:
            crewai_agent: The CrewAI agent to be updated.
        """
        if crewai_agent.name in self.agents:
            self.agents[crewai_agent.name] = crewai_agent
            print(f"Agent '{crewai_agent.name}' has been updated in the MAMA framework.")
        else:
            print(f"Agent '{crewai_agent.name}' not found in the MAMA framework.")

    def remove_agent(self, agent_name: str):
        """
        Remove an agent from the framework by name.

        Args:
            agent_name (str): The name of the agent to be removed.
        """
        if agent_name in self.agents:
            del self.agents[agent_name]
            print(f"Agent '{agent_name}' has been removed from the MAMA framework.")
        else:
            print(f"Agent '{agent_name}' not found in the MAMA framework.")

    def extract_sentiment(self, query: str) -> str:
        """
        Extract the sentiment from the query.

        This is a simplified method and can be expanded with more sophisticated NLP techniques.

        Args:
            query (str): The input query.

        Returns:
            str: The sentiment (positive, negative, neutral, etc.).
        """
        if "good" in query or "happy" in query:
            return "positive"
        elif "bad" in query or "sad" in query:
            return "negative"
        elif "neutral" in query:
            return "neutral"
        elif "not" in query and "happy" in query:
            return "sarcasm"
        else:
            return "neutral"  # Default sentiment

    def process_query(self, query: str):
        """
        Process a query by interacting with the MAMA Registrar to find the best agent.

        Args:
            query (str): The input query to be processed.

        Returns:
            str: The name of the selected agent or a message indicating no suitable agent was found.
        """
        # Extract sentiment from the query
        sentiment = self.extract_sentiment(query)

        # Get a list of agents from the registrar
        best_agent = self.registrar.evaluate_agents(query, sentiment)
        
        # Check if the best agent is suitable for the query
        if best_agent:
            # Get agent details and request the agent to process the query
            agent = self.agents.get(best_agent[0])
            response = agent.receive_request(query)
            
            if response is None:
                print(f"Agent '{best_agent[0]}' is not suitable, finding another agent...")
                # Continue searching for an appropriate agent
                best_agent = self.registrar.find_next_best_agent(query, sentiment)
                if best_agent:
                    agent = self.agents.get(best_agent[0])
                    response = agent.receive_request(query)
                else:
                    print("No other suitable agents available.")
                    return "No suitable agent found."

            return response  # Return the response from the appropriate agent
        else:
            print("No agents available for query.")
            return "No suitable agent"

    def find_next_best_agent(self, query: str, sentiment: str):
        """
        Find the next best agent if the current best agent is not suitable.

        Args:
            query (str): The input query.
            sentiment (str): The sentiment extracted from the query.

        Returns:
            tuple: The best agent's name, address, and port or None if no suitable agent is found.
        """
        next_best_agent = self.registrar.find_next_best_agent(query, sentiment)
        if next_best_agent:
            print(f"Next best agent found: {next_best_agent[0]}")
            return next_best_agent
        else:
            print(f"No more agents available for query: '{query}' with sentiment '{sentiment}'")
            return None
