from .network import receive_message, send_message

class MAMARegistrar:
    """The MAMA Registrar Service that receives PML messages and calculates agent popularity."""

    def __init__(self, port: int):
        self.port = port
        self.agent_scores = {}  # Store agent relevance and popularity scores
        self.agent_query_count = {}  # Track the number of queries processed by each agent

    def listen_for_pml(self):
        """Listen for incoming PML messages."""
        print(f"Registrar listening for PML messages on port {self.port}...")
        while True:
            pml_message = receive_message(self.port)
            self.process_pml(pml_message)

    def process_pml(self, pml_message: dict):
        """Process a received PML message and update agent popularity."""
        agent_name = pml_message['agent_name']
        relevance = pml_message['relevance']

        # Update agent's popularity based on relevance
        if agent_name not in self.agent_scores:
            self.agent_scores[agent_name] = 0.0
            self.agent_query_count[agent_name] = 0

        # Update the total score (relevance) for the agent
        self.agent_scores[agent_name] += relevance
        self.agent_query_count[agent_name] += 1

        # Calculate popularity as the average relevance score per query
        popularity = self.agent_scores[agent_name] / self.agent_query_count[agent_name]
        
        print(f"Updated popularity for {agent_name}: {popularity}")

    def propagate_pml(self, port: int):
        """Propagate PML messages to other registrars (if needed)."""
        send_message(port, {"registrar_data": self.agent_scores})

    def get_popularity(self, agent_name: str) -> float:
        """Get the popularity score of a given agent."""
        if agent_name in self.agent_scores:
            return self.agent_scores[agent_name] / self.agent_query_count[agent_name]
        return 0.0
