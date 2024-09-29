import csv
import yaml
from mama.agent import CrewAIAgent
from mama.mama_framework import MAMAFramework
from mama.registrar import MAMARegistrar
import threading
from datasets import load_dataset

# Start the MAMA Registrar in a separate thread
def start_registrar():
    registrar = MAMARegistrar(port=8089)
    registrar.listen_for_registration()

registrar_thread = threading.Thread(target=start_registrar, daemon=True)
registrar_thread.start()

# Initialize the MAMA framework
mama_framework = MAMAFramework(registrar=MAMARegistrar())

# Load CrewAI agent configurations
with open('MaMaCrew/configs/creawai_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Register all CrewAI agents from the configuration file
agents = []
for agent_config in config['agents']:
    agent = CrewAIAgent(agent_config['path'])
    agents.append(agent)
    mama_framework.add_agent(agent)

# Load SST-2 Dataset from Hugging Face
dataset = load_dataset("glue", "sst2")

# Select the test split for evaluation
test_data = dataset["test"]

# Open CSV file for writing the evaluation results
with open('sst2_mama_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sentence', 'Label', 'Selected Agent', 'Agent Answer'])

    # Iterate over the test dataset
    for example in test_data:
        sentence = example['sentence']
        label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

        # Convert label to a string for comparison
        correct_answer = "positive" if label == 1 else "negative"

        # Process the sentence through the MAMA framework
        selected_agent = mama_framework.process_query(sentence)

        # Write the result to the CSV file
        writer.writerow([sentence, correct_answer, selected_agent, selected_agent])

print("Evaluation completed. Results written to 'sst2_mama_evaluation_results.csv'.")
