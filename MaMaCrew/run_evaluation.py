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

# Select the train split for warm-up and evaluation
train_data = dataset["train"]

# Open CSV file for writing the evaluation results on the training set
with open('sst2_mama_training_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='#')
    writer.writerow(['Question', 'Agentname', 'Answer value', 'Labeled Answer'])

    # Iterate over the training dataset (used for warm-up and evaluation)
    for example in train_data:
        sentence = example['sentence']
        label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

        # Convert label to a string for comparison
        labeled_answer = "positive" if label == 1 else "negative"

        # Process the sentence through the MAMA framework to get the selected agent
        selected_agent = mama_framework.process_query(sentence)

        # Write the result to the CSV file in the requested format
        writer.writerow([f"Question: {sentence}", 
                         f"Agentname: {selected_agent}", 
                         f"Answer value: {selected_agent}", 
                         f"Labeled Answer: {labeled_answer}"])

print("Training evaluation completed. Results written to 'sst2_mama_training_evaluation_results.csv'.")

# Select the test split for final evaluation
test_data = dataset["test"]

# Open CSV file for writing the final evaluation results on the test set
with open('sst2_mama_test_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='#')
    writer.writerow(['Question', 'Agentname', 'Answer value'])

    # Iterate over the test dataset for final evaluation
    for example in test_data:
        sentence = example['sentence']

        # Process the sentence through the MAMA framework to get the selected agent
        selected_agent = mama_framework.process_query(sentence)

        # Write the result to the CSV file in the requested format
        writer.writerow([f"Question: {sentence}", 
                         f"Agentname: {selected_agent}", 
                         f"Answer value: {selected_agent}"])

print("Test evaluation completed. Results written to 'sst2_mama_test_evaluation_results.csv'.")
