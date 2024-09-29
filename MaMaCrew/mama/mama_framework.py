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

    def list_agents(self):
        """
        List all registered agents in the MAMA framework.
        
        Returns:
            list: A list of registered agent names.
        """
        return list(self.agents.keys())

    def process_query(self, query: str):
        """
        Process a query by interacting with the MAMA Registrar to find the best agent.

        Args:
            query (str): The input query to be processed.

        Returns:
            str: The name of the selected agent or a message indicating no suitable agent was found.
        """
        # Extract sentiment markup from the query
        markup = self.extract_markup(query)
        sentiment = self.determine_sentiment(markup)

        # Evaluate agents through the registrar based on the sentiment
        best_agent = self.registrar.evaluate_agents(query, sentiment)
        if best_agent:
            print(f"Best agent for query '{query}': {best_agent[0]} at {best_agent[1]}:{best_agent[2]}")
            return best_agent[0]  # Return the agent name for logging/evaluation
        else:
            print(f"No suitable agent found for query '{query}'")
            return "No suitable agent"

    def extract_markup(self, query: str) -> dict:
        """
        Extract sentiment-related markup from the query. This function determines the sentiment in the query
        and returns a markup dictionary.

        Args:
            query (str): The input query text.

        Returns:
            dict: A dictionary containing sentiment markup.
        """
        # Example markup extraction logic (you can improve this)
        markup = {
            "positive": 1.0 if "good" in query or "happy" in query else 0.0,
            "negative": 1.0 if "bad" in query or "sad" in query else 0.0,
            "sarcasm": 1.0 if "not" in query and "good" in query else 0.0
        }
        return markup

    def determine_sentiment(self, markup: dict) -> str:
        """
        Determine the dominant sentiment from the markup data.

        Args:
            markup (dict): The extracted sentiment markup.

        Returns:
            str: The detected sentiment (e.g., "positive", "negative", "sarcasm").
        """
        if markup.get("positive", 0) > 0:
            return "positive"
        elif markup.get("negative", 0) > 0:
            return "negative"
        elif markup.get("sarcasm", 0) > 0:
            return "sarcasm"
        return "neutral"
