import yaml
from crewai import CrewAgentManager

class Crew:
    def __init__(self, config_file, task_file):
        # Load agent and task configurations
        with open(config_file, 'r') as file:
            self.agent_config = yaml.safe_load(file)
        
        with open(task_file, 'r') as file:
            self.task_config = yaml.safe_load(file)
        
        # Initialize the CREW AI agent manager
        self.manager = CrewAgentManager()
    
    def start(self):
        # Start all agents as per the configuration
        self.manager.setup_agents(self.agent_config)
        
        # Initialize tasks between agents
        self.manager.setup_tasks(self.task_config)
        
        # Start the entire process
        self.manager.run()

    def stop(self):
        # Clean up agents and resources
        self.manager.shutdown()
