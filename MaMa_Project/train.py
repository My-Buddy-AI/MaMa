from mama.sst2_loader import load_sst2
from mama.mama_framework import MAMAFramework

def train_and_evaluate():
    """Train and evaluate agents using the SST-2 dataset."""
    dataset = load_sst2()
    mama = MAMAFramework()

    train_data = dataset['train']
    test_data = dataset['test']

    print("Starting training loop...")

    for sentence in train_data['sentence']:
        prediction = mama.process_query(sentence)
        print(f"Sentence: '{sentence}', Prediction: {prediction}")

    print("\nStarting evaluation on test set...")

    correct_predictions = 0
    for sentence, label in zip(test_data['sentence'], test_data['label']):
        prediction = mama.process_query(sentence)
        correct = (prediction == "positive" and label == 1) or (prediction == "negative" and label == 0)
        if correct:
            correct_predictions += 1
        print(f"Sentence: '{sentence}', Prediction: {prediction}, Actual: {'positive' if label == 1 else 'negative'}")

    accuracy = correct_predictions / len(test_data['sentence'])
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_and_evaluate()
