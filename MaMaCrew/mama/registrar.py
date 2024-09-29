from .network import receive_message

class MAMARegistrar:
    """
    The MAMA Registrar Service that receives agent registrations via PML and dynamically evaluates them.
    Tracks the relevance and popularity of agents based on the received queries.
    """

    def __init__(self, port: int = 8089):
        self.port = port
        self.agent_registry = {}  # Store agents with their relevance and address
        print(f"MAMA Registrar initialized on port {self.port}.")

    def listen_for_registration(self):
        """Listen for incoming PML messages for agent registration."""
        print(f"Registrar listening for PML messages on port {self.port}...")
        while True:
            pml_message = receive_message(self.port)
            self.register_agent(pml_message)

    def register_agent(self, pml_message: dict):
        """Process a received PML message and register the agent."""
        agent_name = pml_message['agent_name']
        relevance_score = pml_message['relevance']
        agent_address = pml_message['address']
        agent_port = pml_message['port']

        # Register or update the agent's details in the registry
        self.agent_registry[agent_name] = {
            'relevance_score': relevance_score,
            'address': agent_address,
            'port': agent_port
        }

        print(f"Registered agent '{agent_name}' at {agent_address}:{agent_port} with relevance score: {relevance_score}")

    def evaluate_agents(self, query: str):
        """Evaluate all registered agents and return the most relevant agent."""
        best_agent = None
        highest_relevance = -1

        for agent_name, data in self.agent_registry.items():
            relevance_score = data['relevance_score']
            if relevance_score > highest_relevance:
                highest_relevance = relevance_score
                best_agent = (agent_name, data['address'], data['port'])

        if best_agent:
            print(f"Best agent selected for query '{query}': {best_agent[0]} with relevance {highest_relevance}")
            return best_agent
        else:
            print(f"No agents available for query '{query}'")
            return None
