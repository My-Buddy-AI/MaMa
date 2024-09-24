import random
from typing import List, Dict

class AIAgent:
    """Base class for AI Agents with profiles and relevance calculation."""
    
    def __init__(self, name: str, profile: Dict[str, float]):
        """
        Initialize an agent with a name and a profile.
        
        Args:
            name (str): The name of the agent.
            profile (Dict[str, float]): Profile describing expertise on different sentiment types.
        """
        self.name = name
        self.profile = profile  # Expertise profile (e.g., {"positive": 0.8, "negative": 0.2})
        self.popularity = 0  # How popular the agent is

    def evaluate(self, query: str, markup: Dict[str, float]) -> float:
        """
        Calculate relevance score based on the agent's profile and the query markup.
        
        Args:
            query (str): The input sentence or query.
            markup (Dict[str, float]): Markup prompt extracted from the query.
        
        Returns:
            float: The relevance score for the agent.
        """
        relevance = 0.0
        for tag, weight in markup.items():
            if tag in self.profile:
                relevance += self.profile[tag] * weight
        
        return relevance
    
    def extract_markup(self, query: str) -> Dict[str, float]:
        """
        Extract markup prompts from the input query.
        
        Args:
            query (str): The input sentence or query.
        
        Returns:
            Dict[str, float]: Markup prompts extracted from the query.
        """
        # Simple example: weight the query based on sentiment-related terms
        markup = {
            "positive": 0.8 if "good" in query or "happy" in query else 0.2,
            "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1
        }
        return markup

class PositiveClassifier(AIAgent):
    def __init__(self):
        profile = {"positive": 1.0, "negative": 0.2, "sarcasm": 0.1}
        super().__init__("Positive Classifier", profile)

class NegativeClassifier(AIAgent):
    def __init__(self):
        profile = {"positive": 0.2, "negative": 1.0, "sarcasm": 0.1}
        super().__init__("Negative Classifier", profile)

# Add other agents similarly

class OverallSentimentAggregator(AIAgent):
    def __init__(self, agents: List[AIAgent]):
        super().__init__("Overall Sentiment Aggregator", {"positive": 0.5, "negative": 0.5})
        self.agents = agents

    def aggregate(self, query: str, markup: Dict[str, float]) -> str:
        """Aggregate the results from different agents to provide a final classification."""
        results = []
        for agent in self.agents:
            relevance = agent.evaluate(query, markup)
            results.append((agent.name, relevance))
        
        # Return the agent with the highest relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        best_agent = results[0][0]
        
        return f"Selected Agent: {best_agent}"
