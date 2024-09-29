import csv
import asyncio
import yaml
from mama.agent import CrewAIAgent
from mama.mama_framework import MAMAFramework
from mama.registrar import MAMARegistrar
from datasets import load_dataset
from sklearn.metrics import accuracy_score


async def start_registrar():
    """
    Start the MAMA Registrar to listen for agent registrations.
    """
    registrar = MAMARegistrar(port=8089)
    await registrar.listen_for_registration()

async def train_and_evaluate():
    """
    Train and evaluate agents using the SST-2 dataset.
    """
    # Start the MAMA Registrar in a separate task
    registrar_task = asyncio.create_task(start_registrar())

    # Initialize the MAMA framework
    registrar = MAMARegistrar()
    mama_framework = MAMAFramework(registrar)

    # Load CrewAI agent configurations
    with open('MaMaCrew/configs/creawai_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Register all CrewAI agents from the configuration file (in training mode)
    agents = []
    for agent_config in config['agents']:
        agent = CrewAIAgent(agent_config['path'], training_mode=True)
        agents.append(agent)
        mama_framework.add_agent(agent)

    # Load SST-2 Dataset from Hugging Face
    dataset = load_dataset("glue", "sst2")

    # Training Phase: Train agents on the training set
    train_data = dataset["train"]
    
    print("Training agents...")

    for example in train_data:
        sentence = example['sentence']
        label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)
        
        # Convert label to string for training
        sentiment = "positive" if label == 1 else "negative"
        query = sentence + (" good" if label == 1 else " bad")  # Help agents detect sentiment

        # Send the query to the MAMA framework for training
        await mama_framework.process_query(query, sentiment)

    print("Training phase completed.")

    # Evaluation Phase: Evaluate on test set
    test_data = dataset["test"]
    y_true = []
    y_pred = []
    results = []

    print("Evaluating on test set...")

    for example in test_data:
        sentence = example['sentence']
        label = example['label']  # SST-2 labels: 1 (positive), 0 (negative)
        
        # Convert label to string for evaluation
        correct_answer = "positive" if label == 1 else "negative"
        y_true.append(correct_answer)

        # Process the sentence through the MAMA framework
        selected_agent = await mama_framework.process_query(sentence, correct_answer)

        # The predicted answer is the selected agent's response (either "positive" or "negative")
        y_pred.append(selected_agent)

        # Collect results for CSV
        results.append([sentence, correct_answer, selected_agent])

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
            sentiment = "positive" if label == 1 else "negative"
            query = sentence + (" good" if label == 1 else " bad")
            await mama_framework.process_query(query, sentiment)

        # Re-evaluate after retraining
        y_pred = []
        for example in test_data:
            sentence = example['sentence']
            selected_agent = await mama_framework.process_query(sentence, correct_answer)
            y_pred.append(selected_agent)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy after retraining: {accuracy * 100:.2f}%")

    # Write results to CSV
    with open('sst2_mama_evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Sentence', 'Correct Answer', 'Selected Agent'])
        for result in results:
            writer.writerow(result)

    print("Evaluation completed. Results written to 'sst2_mama_evaluation_results.csv'.")

    # Cancel the registrar task
    registrar_task.cancel()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(train_and_evaluate())
    else:
        asyncio.run(train_and_evaluate())
