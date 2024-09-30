from .registrar import MAMARegistrar
from .agent import CrewAIAgent
from .network import receive_message, send_message
import time

class MAMAFramework:
    """
    MAMAFramework manages dynamically added CrewAI agents and processes queries by interacting with MAMA Registrar.
    It discovers the most relevant agent for each query and allows dynamic addition, updating, and removal of agents.
    """

    def __init__(self, registrar: MAMARegistrar):
        """Initialize the MAMA framework with a reference to the MAMA Registrar."""
        self.registrar = registrar
        self.agents = {}  # Dictionary to store agents by name

    def add_agent(self, crewai_agent: CrewAIAgent):
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

    def update_agent(self, crewai_agent: CrewAIAgent):
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

    def training(self, query: str, sentiment: str):
            """
            Process a query by interacting with the MAMA Registrar to find the best agent.
            
            Args:
                query (str): The input query to be processed.
                sentiment (str): The sentiment type for query classification (e.g., 'positive', 'negative', 'sarcasm').

            Returns:
                str: The name of the selected agent or a message indicating no suitable agent was found.
            """
            print(f"Processing query '{query}' with sentiment '{sentiment}'...")

            # Interact with the registrar to find the best agent (synchronously)
            best_agent = self.registrar.training(query,sentiment)

            if best_agent:
                agent_name, agent_address, agent_port = best_agent
                print(f"Best agent for query '{query}' is '{agent_name}' at {agent_address}:{agent_port}.")
                
                # Send the query to the selected agent synchronously
                self.send_query_to_agent(agent_name, agent_address, agent_port, query)
                return agent_name
            else:
                #print(f"No suitable agent found for query '{query}' with sentiment '{sentiment}'.")
                return "No suitable agent"
        
    def process_query(self, query: str, sentiment: str):
            """
            Process a query by interacting with the MAMA Registrar to find the best agent.
            
            Args:
                query (str): The input query to be processed.
                sentiment (str): The sentiment type for query classification (e.g., 'positive', 'negative', 'sarcasm').

            Returns:
                str: The name of the selected agent or a message indicating no suitable agent was found.
            """
            print(f"Processing query '{query}' with sentiment '{sentiment}'...")

            # Interact with the registrar to find the best agent (synchronously)
            best_agent = self.registrar.evaluate_agents(query)

            if best_agent:
                agent_name, agent_address, agent_port = best_agent
                print(f"Best agent for query '{query}' is '{agent_name}' at {agent_address}:{agent_port}.")
                
                # Send the query to the selected agent synchronously
                self.send_query_to_agent(agent_name, agent_address, agent_port, query)
                return agent_name
            else:
                #print(f"No suitable agent found for query '{query}' with sentiment '{sentiment}'.")
                return "No suitable agent"

    def send_query_to_agent(self, agent_name: str, agent_address: str, agent_port: int, query: str):
        """
        Send the query to the selected agent with retry logic.

        Args:
            agent_name (str): The name of the selected agent.
            agent_address (str): The address of the agent.
            agent_port (int): The port of the agent.
            query (str): The query to be processed by the agent.
        """
        print(f"Sending query '{query}' to agent '{agent_name}' at {agent_address}:{agent_port}...")

        # Sending query using the agent's communication mechanism
        message = {"query": query}
        
        retries = 0
        max_retries = 10
        wait_time = 0.1  # 100ms

        while retries < max_retries:
            try:
                send_message(agent_port, message)  # Synchronous message sending
                print(f"Query successfully sent to agent '{agent_name}' at {agent_address}:{agent_port}")
                return  # Exit the function once the message is successfully sent
            except Exception as e:
                retries += 1
                print(f"Failed to send query to agent '{agent_name}' at {agent_address}:{agent_port}. Attempt {retries}/{max_retries}. Error: {e}")
                
                if retries < max_retries:
                    print(f"Retrying in {wait_time * 1000:.0f}ms...")
                    time.sleep(wait_time)  # Synchronous wait
                else:
                    print(f"Exceeded max retries. Could not send query to agent '{agent_name}' at {agent_address}:{agent_port}.")
                    return
