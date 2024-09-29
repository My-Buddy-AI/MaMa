import random
from typing import Dict, List
from .pml import PMLMessage
from .network import send_message


class AIAgent:
    """Base class for AI Agents with profiles, ports, and relevance calculation."""

    def __init__(self, name: str, profile: Dict[str, float]):
        """
        Initialize the agent with dynamically assigned ports and a profile.
        
        Args:
            name (str): The name of the agent.
            profile (Dict[str, float]): Profile describing expertise on different sentiment types.
        """
        self.name = name
        self.profile = profile
        self.popularity = 0

        # Dynamically assign ports
        self.receive_port = self.assign_dynamic_port()
        self.reply_port = self.assign_dynamic_port()
        self.pml_port = self.assign_dynamic_port()

        print(f"Agent {self.name} initialized with ports: receive={self.receive_port}, reply={self.reply_port}, PML={self.pml_port}")

    def assign_dynamic_port(self):
        """Assign an available port dynamically."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', 0))  # Bind to an available port
        port = s.getsockname()[1]  # Get the dynamically assigned port
        s.close()  # Release the socket
        return port

    def receive_request(self, query: str):
        """Receive a query, process it, and send the PML message."""
        print(f"Agent {self.name} received query: {query}")
        
        # Extract markup prompts from the query
        markup = self.extract_markup(query)

        # Calculate relevance score based on the agent's profile and the markup
        relevance = self.evaluate(query, markup)
        
        # Generate a result based on the query and markup
        result = self.process_query(query, markup)

        # Send the result back to the client
        self.send_reply(result)

        # Send the PML message with the relevance score to the Registrar
        self.send_pml(query, result, relevance)

    def process_query(self, query: str, markup: Dict[str, float]) -> str:
        """Process the query based on the content and return a response."""
        if "good" in query:
            return "positive"
        elif "bad" in query:
            return "negative"
        return "neutral"

    def evaluate(self, query: str, markup: Dict[str, float]) -> float:
        """Calculate relevance score based on the agent's profile and the query markup."""
        relevance = 0.0
        for tag, weight in markup.items():
            if tag in self.profile:
                relevance += self.profile[tag] * weight
        return relevance

    def extract_markup(self, query: str) -> Dict[str, float]:
        """Extract markup prompts from the input query."""
        markup = {
            "positive": 0.8 if "good" in query or "happy" in query else 0.2,
            "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1
        }
        return markup

    def send_reply(self, result: str):
        """Send the reply to the client."""
        print(f"Agent {self.name} sending reply: {result}")
        send_message(self.reply_port, {"agent": self.name, "result": result})

    def send_pml(self, query: str, result: str, relevance: float):
        """Send a PML (Prompt Markup Language) message to the Registrar."""
        pml_message = PMLMessage(agent_name=self.name, query=query, result=result, relevance=relevance, agent_port=self.reply_port)
        print(f"Agent {self.name} sending PML: {pml_message}")
        send_message(self.pml_port, pml_message.to_dict())


class PositiveClassifier(AIAgent):
    def __init__(self):
        profile = {"positive": 1.0, "negative": 0.2, "sarcasm": 0.1}
        super().__init__("Positive Classifier", profile)


class NegativeClassifier(AIAgent):
    def __init__(self):
        profile = {"positive": 0.2, "negative": 1.0, "sarcasm": 0.1}
        super().__init__("Negative Classifier", profile)


class OverallSentimentAggregator:
    """Aggregates results from multiple agents and provides a final sentiment classification."""

    def __init__(self, agents: List[AIAgent]):
        """
        Initialize the aggregator with a list of agents.
        
        Args:
            agents (List[AIAgent]): A list of agent instances (e.g., PositiveClassifier, NegativeClassifier).
        """
        self.agents = agents

    def aggregate(self, query: str) -> str:
        """Aggregate results from different agents and return the best prediction."""
        results = []
        markup = self.extract_markup(query)

        for agent in self.agents:
            relevance = agent.evaluate(query, markup)
            result = agent.process_query(query, markup)
            results.append((agent.name, result, relevance))

        # Sort results by relevance score and return the most relevant agent's result
        results.sort(key=lambda x: x[2], reverse=True)
        best_agent, best_result, best_relevance = results[0]
        print(f"Selected agent: {best_agent} with relevance {best_relevance}")
        return best_result

    def extract_markup(self, query: str) -> Dict[str, float]:
        """Extract markup prompts from the input query."""
        markup = {
            "positive": 0.8 if "good" in query or "happy" in query else 0.2,
            "negative": 0.8 if "bad" in query or "sad" in query else 0.2,
            "sarcasm": 0.9 if "not" in query and "happy" in query else 0.1
        }
        return markup
