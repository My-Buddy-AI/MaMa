from .agent import PositiveClassifier, NegativeClassifier, OverallSentimentAggregator
from .reinforcement_learning import MultiAgentReinforcementLearning
from .markup_extraction import extract_markup

class MAMAFramework:
    def __init__(self):
        """Initialize MAMA framework with a set of agents."""
        self.agents = [
            PositiveClassifier(),
            NegativeClassifier(),
            # Add other agents
        ]
        self.aggregator = OverallSentimentAggregator(self.agents)
        self.rl_system = MultiAgentReinforcementLearning([agent.name for agent in self.agents])

    def process_query(self, query: str) -> str:
        """Process a query using the most relevant agents."""
        markup = extract_markup(query)  # Extract markup from query
        relevant_agents = self.select_agents(query, markup)
        final_result = self.aggregator.aggregate(query, markup)
        return final_result

    def select_agents(self, query: str, markup: Dict[str, float]):
        """Select relevant agents based on the query markup."""
        selected_agents = []
        for agent in self.agents:
            relevance = agent.evaluate(query, markup)
            if relevance > 0.5:  # Relevance threshold
                selected_agents.append(agent)
        return selected_agents
