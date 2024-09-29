import csv
from mama.agent import CrewAIAgent
from mama.mama_framework import MAMAFramework
from mama.registrar import MAMARegistrar
import threading

# Start the MAMA Registrar in a separate thread
def start_registrar():
    registrar = MAMARegistrar(port=8089)
    registrar.listen_for_registration()

registrar_thread = threading.Thread(target=start_registrar, daemon=True)
registrar_thread.start()

# Initialize the MAMA framework
mama_framework = MAMAFramework(registrar=MAMARegistrar())

# Define and register agents
agents = [
    CrewAIAgent("Positive Work", {"positive": 1.0, "negative": 0.2}),
    CrewAIAgent("Negative Work", {"positive": 0.2, "negative": 1.0}),
    # Add more agents for different contexts here
]

# Add agents to the MAMA framework
for agent in agents:
    mama_framework.add_agent(agent)

# Example queries
queries = [
    "I love working here!",
    "This project is a disaster."
]

# Run evaluation and generate CSV output
with open('mama_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Query', 'Selected Agent'])

    for query in queries:
        selected_agent = mama_framework.process_query(query)
        writer.writerow([query, selected_agent])

print("Evaluation completed. Results written to 'mama_evaluation_results.csv'.")
