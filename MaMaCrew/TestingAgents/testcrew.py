# Import necessary modules from CrewAI
from crewai import Agent, Task, Crew, Process
from langchain.chat_models import ChatOpenAI  # For OpenAI GPT models

# Define your OpenAI API key (Ensure this is set in your environment)
import os
os.environ["OPENAI_API_KEY"] = "sk-proj-tVj5BU9qy6FR40cMAci1T3BlbkFJ0k34pfmpFiQTeTyUY7Cd"  # Replace with your actual key

# Define the Mischievousness Sentiment Agent with OpenAI's GPT model
mischief_agent = Agent(
    role='Mischievousness Sentiment Agent',
    goal='Detect playful mischievousness, sarcasm, and humorous rebellion in language',
    backstory="""
        This agent specializes in identifying playful mischievousness, light sarcasm, and humorous rebellion in language.
        It recognizes phrases that involve joking or subtle avoidance of authority, particularly in personal or family contexts.
        The agent detects language patterns that involve hiding or sneaking behavior, particularly in familial or authority-driven contexts.
    """,
    verbose=True,
    # Specify the GPT model (using GPT-4 here)
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.7)  # Use gpt-4 or gpt-3.5-turbo
)

# Example phrases from the YAML content
example_phrases = [
    "Hide new secretions from the parental units",
    "Avoid telling the boss about the accidental spill",
    "Sneak the cookies before mom notices",
    "Pretend the project was finished on time"
]

# Define a task for the agent to analyze phrases
task = Task(
    description="Analyze text to detect mischievous, sarcastic, and humorous rebellion in statements.",
    expected_output="Identify tone, humor, and mild rebelliousness in language.",
    agent=mischief_agent
)

# Create the Crew that will manage the task with the agent
crew = Crew(
    agents=[mischief_agent],
    tasks=[task],
    verbose=True,
    process=Process.sequential
)

# Kick off the crew to start the task
result = crew.kickoff()

# Print the result after analyzing the phrases
print("### Analysis Results ###")
for phrase in example_phrases:
    print(f"Analyzing phrase: {phrase}")
    print(result)
