import csv
import yaml
from mama.agent import CrewAIAgent
from mama.mama_framework import MAMAFramework
from mama.registrar import MAMARegistrar
import threading
from datasets import load_dataset
from sklearn.metrics import accuracy_score

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

# Register all CrewAI agents from the configuration file (in training mode)
agents = []
for agent_config in config['agents']:
    agent = CrewAIAgent(agent_config['path'], training_mode=True)  # Training mode is set to True
    agents.append(agent)
    mama_framework.add_agent(agent)

# Load SST-2 Dataset from Hugging Face
dataset = load_dataset("glue", "sst2")

# Training Phase: Train agents on the training set
train_data = dataset["train"]

# Train agents using the training set (train all positive examples for positive agent, etc.)
for example in train_data:
    sentence = example['sentence']
    label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

    # Convert label to string for training
    query = sentence
    if label == 1:
        query += " good"  # Ensure the agent knows it's a positive sentiment
    else:
        query += " bad"  # Ensure the agent knows it's a negative sentiment

    # Send the query to the MAMA framework for training
    mama_framework.process_query(query)

print("Training phase completed.")

# Evaluation Phase: Evaluate on test set
test_data = dataset["test"]
y_true = []
y_pred = []

for example in test_data:
    sentence = example['sentence']
    label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

    # Convert label to string for evaluation
    correct_answer = "positive" if label == 1 else "negative"
    y_true.append(correct_answer)

    # Process the sentence through the MAMA framework
    selected_agent = mama_framework.process_query(sentence)

    # The predicted answer is the selected agent's response (either "positive" or "negative")
    y_pred.append(selected_agent)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test set accuracy: {accuracy * 100:.2f}%")

# Retrain if accuracy is below threshold
if accuracy < 0.80:  # Threshold for retraining
    print("Accuracy below threshold, retraining agents...")
    # Re-run the training phase
    for example in train_data:
        sentence = example['sentence']
        label = example['label']
        query = sentence
        if label == 1:
            query += " good"
        else:
            query += " bad"
        mama_framework.process_query(query)

    # Re-evaluate after retraining
    y_pred = []
    for example in test_data:
        sentence = example['sentence']
        selected_agent = mama_framework.process_query(sentence)
        y_pred.append(selected_agent)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy after retraining: {accuracy * 100:.2f}%")

# Final evaluation completed.
print("Final evaluation completed.")
