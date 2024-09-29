from .registrar import MAMARegistrar

class MAMAFramework:
    """
    MAMAFramework manages dynamically added CrewAI agents and processes queries by interacting with MAMA Registrar.
    It discovers the most relevant agent for each query.
    """

    def __init__(self, registrar: MAMARegistrar):
        """Initialize the MAMA framework with a reference to the MAMA Registrar."""
        self.registrar = registrar

    def process_query(self, query: str):
        """Process a query by interacting with the MAMA Registrar to find the best agent."""
        best_agent = self.registrar.evaluate_agents(query)
        if best_agent:
            print(f"Best agent for query '{query}': {best_agent[0]} at {best_agent[1]}:{best_agent[2]}")
            return best_agent[0]  # Return the agent name for logging/evaluation
        else:
            print(f"No suitable agent found for query '{query}'")
            return "No suitable agent"
