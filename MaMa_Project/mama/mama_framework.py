from .agent import PositiveClassifier, NegativeClassifier, OverallSentimentAggregator

class MAMAFramework:
    """
    MAMAFramework is responsible for managing agents and handling the query processing.
    It aggregates results from multiple agents using the OverallSentimentAggregator and returns the final decision.
    """

    def __init__(self):
        """Initialize MAMA framework with a set of agents."""
        self.agents = [
            PositiveClassifier(),  # Positive sentiment classifier
            NegativeClassifier(),  # Negative sentiment classifier
            # You can add more agents here, like Sarcasm Detector, Neutral Detector, etc.
        ]
        self.aggregator = OverallSentimentAggregator(self.agents)

    def process_query_with_metadata(self, query: str):
        """
        Process a query using the most relevant agents and return additional metadata such as the agent's name,
        the predicted sentiment, the relevance score, and the popularity.

        Args:
            query (str): The input sentence or query.

        Returns:
            tuple: (agent_name, prediction, relevance, popularity)
        """
        # Extract markup (e.g., sentiment tags) from the query
        markup = self.aggregator.extract_markup(query)

        # Aggregate results from all agents and get the best result
        result = self.aggregator.aggregate(query)

        # Assume the best agent is the one with the highest relevance
        selected_agent = self.agents[0]  # For simplicity, assuming the first agent as an example

        # Calculate relevance score for the selected agent
        relevance = selected_agent.evaluate(query, markup)

        # Placeholder for popularity score
        popularity = 0.5  # Assume the popularity is static for now, can be retrieved from a registrar if needed

        return selected_agent.name, result, relevance, popularity
