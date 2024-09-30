# Import necessary modules from CrewAI
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI  # For OpenAI GPT models

# Define your OpenAI API key (Ensure this is set in your environment)
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-tVj5BU9qy6FR40cMAci1T3BlbkFJ0k34pfmpFiQTeTyUY7Cd"  # Replace with your actual key


# Function to load agents from a folder containing YAML files
def load_agents_from_folder(folder_path):
    agents = []
    # Check for YAML files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.yaml'):
            # Load the YAML content
            with open(os.path.join(folder_path, filename), 'r') as file:
                agent_data = yaml.safe_load(file)
                
                # Extracting data from the YAML file
                agent_name = agent_data.get('name', 'Unnamed Agent')
                role = agent_data.get('specialty', 'No role defined')  # Assuming 'specialty' is the role
                backstory = agent_data.get('behavior', 'No backstory provided')  # Assuming 'behavior' is backstory
                
                # Create an Agent object using the extracted data
                agent = Agent(
                    role=role,
                    goal=f"{role} - This agent specializes in detecting specific behaviors.",
                    backstory=backstory,
                    verbose=True,
                    llm=ChatOpenAI(model_name="gpt-4", temperature=0.7)  # Specify the GPT model
                )
                agents.append(agent)
    return agents

# Folder path where the agent YAML files are stored
agents_folder_path = '../agents/'  # Replace with your actual folder path

# Load agents dynamically from the folder
loaded_agents = load_agents_from_folder(agents_folder_path)

# Example phrases from the YAML content
example_phrases = [
    "Hide new secretions from the parental units",
    "Avoid telling the boss about the accidental spill",
    "Sneak the cookies before mom notices",
    "Pretend the project was finished on time"
]

# Process each phrase with each loaded agent
for agent in loaded_agents:
    print(f"### Analyzing phrases with agent: {agent.role} ###")
    
    # Create tasks for each phrase and run them
    for phrase in example_phrases:
        # Define a unique task for each phrase
        task = Task(
            description=f"Analyze the phrase: {phrase}",
            expected_output="Detect mischievousness, sarcasm, and rebellious behavior.",
            agent=agent
        )
        
        # Create a crew for this task
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        
        # Kick off the crew to start the task
        result = crew.kickoff()
        
        # Print the result after analyzing the phrase
        print("### Analysis Results ###")
        print(f"Phrase: {phrase}")
        print(result)