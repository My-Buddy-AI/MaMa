from .agent import PositiveClassifier, NegativeClassifier, OverallSentimentAggregator
from .network import send_message

class MAMAFramework:
    def __init__(self):
        """Initialize MAMA framework with a set of agents."""
        self.agents = [
            PositiveClassifier(receive_port=5001, reply_port=5002, pml_port=5003),
            NegativeClassifier(receive_port=5004, reply_port=5005, pml_port=5006),
            # Add other agents here (e.g., Sarcasm Detector, Neutral Detector, etc.)
        ]
        self.aggregator = OverallSentimentAggregator(self.agents)

    def process_query_with_metadata(self, query: str):
        """
        Process a query using the most relevant agents and return additional metadata.
        
        Args:
            query (str): The input sentence or query.

        Returns:
            tuple: (agent_name, prediction, relevance, popularity)
        """
        # Extract markup (e.g., sentiment tags) from the query
        markup = self.aggregator.extract_markup(query)

        # Select the most relevant agent
        selected_agent, relevance = self.select_most_relevant_agent(query, markup)

        # Get the prediction from the selected agent
        prediction = selected_agent.process_query(query, markup)

        # Retrieve the current popularity of the selected agent
        popularity = self.get_agent_popularity(selected_agent)

        # Send the PML data to the registrar
        self.send_pml_to_registrar(selected_agent.name, query, prediction, relevance)

        return selected_agent.name, prediction, relevance, popularity

    def select_most_relevant_agent(self, query: str, markup: dict):
        """Select the agent with the highest relevance score for the given query."""
        best_agent = None
        highest_relevance = -float('inf')

        for agent in self.agents:
            relevance = agent.evaluate(query, markup)
            if relevance > highest_relevance:
                highest_relevance = relevance
                best_agent = agent

        return best_agent, highest_relevance

    def get_agent_popularity(self, agent):
        """Get the popularity score of an agent from the registrar."""
        # Here we could simulate the popularity calculation from the Registrar Service
        # This could also be updated based on actual registrar responses
        return 0.5  # Placeholder popularity score

    def send_pml_to_registrar(self, agent_name, query, prediction, relevance):
        """Send PML (Prompt Markup Language) data to the Registrar Service."""
        pml_data = {
            'agent_name': agent_name,
            'query': query,
            'prediction': prediction,
            'relevance': relevance
        }
        # Assume the registrar is listening on port 5003
        send_message(5003, pml_data)
