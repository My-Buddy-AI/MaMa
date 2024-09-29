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

# Load multiple CrewAI agent configurations from a directory or a list of config files
config_files = [
    'MaMaCrew/agents/positive_agent.yaml',
    'MaMaCrew/agents/negative_agent.yaml',
    'MaMaCrew/agents/sarcasm_agent.yaml',
    'MaMaCrew/agents/neutral_agent.yaml'
]

agents = []
for config_path in config_files:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        agent = CrewAIAgent(config_path, training_mode=True)  # Initialize the agent in training mode
        agents.append(agent)
        mama_framework.add_agent(agent)  # Add the agent to the MAMA framework

# Load SST-2 Dataset from Hugging Face
dataset = load_dataset("glue", "sst2")

### Training Phase: Train agents on the training set
train_data = dataset["train"]

# Open CSV file for writing the training evaluation results
with open('sst2_mama_training_evaluation_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='#')
    writer.writerow(['Question', 'Agentname', 'Answer value', 'Labeled Answer'])

    # Train agents using the training set
    for example in train_data:
        sentence = example['sentence']
        label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

        # Send the query to the appropriate agent based on the label
        query = sentence
        labeled_answer = "positive" if label == 1 else "negative"
        
        if label == 1:
            query += " good"  # Positive query for training
            selected_agent = mama_framework.process_query(query)
        else:
            query += " bad"  # Negative query for training
            selected_agent = mama_framework.process_query(query)

        # Write training result to the CSV file
        writer.writerow([f"Question: {sentence}", 
                         f"Agentname: {selected_agent}", 
                         f"Answer value: {selected_agent}", 
                         f"Labeled Answer: {labeled_answer}"])

print("Training phase completed. Results written to 'sst2_mama_training_evaluation_results.csv'.")

### Evaluation Phase: Evaluate on the evaluation set (which we can simulate from the test set)
validation_data = dataset["validation"] if "validation" in dataset else dataset["test"]  # In case SST-2 doesn't have validation split
y_true = []
y_pred = []

# Validate agents using the validation set
for example in validation_data:
    sentence = example['sentence']
    label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)

    # Convert label to string for evaluation
    correct_answer = "positive" if label == 1 else "negative"
    y_true.append(correct_answer)

    # Process the sentence through the MAMA framework
    selected_agent = mama_framework.process_query(sentence)

    # The predicted answer is the selected agent's response (either "positive" or "negative")
    y_pred.append(selected_agent)

# Calculate accuracy on validation set
validation_accuracy = accuracy_score(y_true, y_pred)
print(f"Validation set accuracy: {validation_accuracy * 100:.2f}%")

# Retrain if validation accuracy is below threshold
if validation_accuracy < 0.80:  # Threshold for retraining
    print("Validation accuracy below threshold, retraining agents...")

    # Re-run the training phase, including sarcasm agent retraining
    for example in train_data:
        sentence = example['sentence']
        label = example['label']
        query = sentence
        if label == 1:
            query += " good"  # Retrain positive agent
        else:
            query += " bad"  # Retrain negative agent
        mama_framework.process_query(query)

    # Re-run validation after retraining
    y_pred = []
    for example in validation_data:
        sentence = example['sentence']
        selected_agent = mama_framework.process_query(sentence)
        y_pred.append(selected_agent)

    validation_accuracy = accuracy_score(y_true, y_pred)
    print(f"Validation accuracy after retraining: {validation_accuracy * 100:.2f}%")

# Only proceed to the test set if validation is successful
if validation_accuracy >= 0.80:
    ### Test Phase: Evaluate on the test set
    test_data = dataset["test"]
    y_true = []
    y_pred = []

    # Open CSV file for writing the test evaluation results
    with open('sst2_mama_test_evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='#')
        writer.writerow(['Question', 'Agentname', 'Answer value'])

        # Iterate over the test dataset for evaluation
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

            # Write test result to the CSV file
            writer.writerow([f"Question: {sentence}", 
                             f"Agentname: {selected_agent}", 
                             f"Answer value: {selected_agent}"])

    # Calculate final accuracy on the test set
    test_accuracy = accuracy_score(y_true, y_pred)
    print(f"Test set accuracy: {test_accuracy * 100:.2f}%")

    # Final evaluation completed.
    print("Final evaluation completed. Test results written to 'sst2_mama_test_evaluation_results.csv'.")
else:
    print("Validation failed, did not proceed to test phase.")
