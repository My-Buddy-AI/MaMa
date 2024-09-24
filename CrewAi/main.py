from crew import Crew
from configparser import ConfigParser

def main():
    # Load configuration files
    crew = Crew(config_file="config/agents.yaml", task_file="config/tasks.yaml")
    
    # Start the CREW AI process
    crew.start()

if __name__ == "__main__":
    main()
