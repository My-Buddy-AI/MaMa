from .agent import CrewAIAgent, OverallSentimentAggregator

class MAMAFramework:
    """
    MAMAFramework is responsible for managing dynamically added CrewAI agents and handling query processing.
    It aggregates results from multiple agents using the OverallSentimentAggregator and returns the final decision.
    """

    def __init__(self):
        """Initialize MAMA framework with a dynamic list of agents."""
        self.agents = []  # Initialize an empty list for agents
        self.aggregator = None  # The aggregator will be initialized once agents are added

    def add_agent(self, crewai_agent: CrewAIAgent):
        """
        Add a new CrewAI agent to the framework, and transform it into a MAMA agent.

        Args:
            crewai_agent (CrewAIAgent): The CrewAI agent to be added and transformed.
        """
        # Transform the agent to MAMA-enabled agent
        crewai_agent.transform_to_mama_agent()

        # Add the transformed agent to the list of MAMA agents
        self.agents.append(crewai_agent)

        # Re-initialize the aggregator with the updated list of agents
        self.aggregator = OverallSentimentAggregator(self.agents)

        print(f"Agent '{crewai_agent.name}' has been added to the MAMA framework.")

    def process_query_with_metadata(self, query: str):
        """
        Process a query using the most relevant agents and return additional metadata such as the agent's name,
        the predicted sentiment, the relevance score, and the popularity.

        Args:
            query (str): The input sentence or query.

        Returns:
            tuple: (agent_name, prediction, relevance, popularity)
        """
        if not self.aggregator:
            raise Exception("No agents are available in the MAMA framework to process the query.")

        # Extract markup from the query and aggregate results from all agents
        result = self.aggregator.aggregate(query)

        # Assume the best agent is the one with the highest relevance
        selected_agent = self.agents[0]  # For simplicity, assuming the first agent as an example

        # Calculate relevance score for the selected agent
        relevance = selected_agent.evaluate(query, selected_agent.extract_markup(query))

        # Placeholder for popularity score
        popularity = 0.5  # This can be dynamically calculated based on usage

        return selected_agent.name, result, relevance, popularity
