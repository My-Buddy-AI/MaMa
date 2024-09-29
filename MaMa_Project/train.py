import csv
import threading
from mama.sst2_loader import load_sst2
from mama.mama_framework import MAMAFramework
from mama.registrar import MAMARegistrar

def evaluate_agents_with_sst2():
    """Evaluate agents using the SST-2 dataset and produce CSV output with detailed evaluation metrics."""
    
    # Load SST-2 dataset
    dataset = load_sst2()
    
    # Initialize the MAMA Framework with all agents
    mama = MAMAFramework()

    # Data splits
    train_data = dataset['train']
    test_data = dataset['test']

    # Open CSV file for writing evaluation results
    with open('mama_evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Query', 'Expected', 'Agent', 'Predicted', 'Correct', 'Relevance', 'Agent Popularity'])

        correct_predictions = 0
        total_predictions = 0

        # Process each sentence in the test set
        for sentence, label in zip(test_data['sentence'], test_data['label']):
            # Expected result (from SST-2 labels)
            expected = 'positive' if label == 1 else 'negative'

            # Process the sentence through MAMA framework
            agent_name, prediction, relevance, popularity = mama.process_query_with_metadata(sentence)

            # Determine if the prediction is correct
            correct = prediction == expected
            if correct:
                correct_predictions += 1
            total_predictions += 1

            # Write the results to the CSV
            writer.writerow([sentence, expected, agent_name, prediction, correct, relevance, popularity])

    # Calculate overall accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Overall Evaluation Accuracy: {accuracy * 100:.2f}%")
    print(f"Results written to 'mama_evaluation_results.csv'.")

def run_registrar():
    """Run the MAMA registrar."""
    registrar = MAMARegistrar(port=5003)  # This registrar listens on PML port 5003
    registrar.listen_for_pml()

if __name__ == "__main__":
    # Initialize and run the MAMA Registrar in a separate thread
    threading.Thread(target=run_registrar).start()

    # Start evaluating the MAMA framework using the SST-2 dataset
    evaluate_agents_with_sst2()
