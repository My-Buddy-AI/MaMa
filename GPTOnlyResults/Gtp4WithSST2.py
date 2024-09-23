import openai
from datasets import load_dataset

# Set your OpenAI API key
openai.api_key = 'apikey'

# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")
print(dataset['test'][0])  

# Function to generate a question for GPT-4 based on the sentiment task
def generate_question(sentence):
    return f"What is the sentiment of the following sentence: '{sentence}'?"

# Function to query GPT-4 using OpenAI API
def query_gpt4(question):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use GPT-4 model
        messages=[
            {"role": "system", "content": "You are a sentiment analysis assistant."},
            {"role": "user", "content": question},
        ],
        max_tokens=100
    )
    return response['choices'][0]['message']['content']

# Function to interpret GPT-4 response and map to labels (0 = negative, 1 = positive)
def interpret_gpt4_response(response):
    if 'positive' in response.lower():
        return 1
    elif 'negative' in response.lower():
        return 0
    else:
        return None  # For unclear responses

# Initialize counters for comparison
correct_predictions = 0
total_predictions = 0

# Process a subset of examples from the SST-2 dataset (test set)
for example in dataset['test']:  # Limiting to 50 examples for faster execution
    print(type(example))  # This will help verify if it's a string or a dictionary
    print(example)  # Print the content of `example`
    sentence = example['sentence']
    true_label = example['label']  # 0 = negative, 1 = positive

    # Generate question based on the SST-2 sentence
    question = generate_question(sentence)
    
    # Query GPT-4
    gpt_response = query_gpt4(question)
    
    # Interpret GPT-4 response and map to sentiment label
    gpt_label = interpret_gpt4_response(gpt_response)

    # Output the results
    print(f"Sentence: {sentence}")
    print(f"Actual Sentiment: {'Positive' if true_label == 1 else 'Negative'}")
    print(f"GPT-4 Response: {gpt_response}")
    print(f"GPT-4 Predicted Sentiment: {'Positive' if gpt_label == 1 else 'Negative' if gpt_label == 0 else 'Unclear'}")
    print("--------------------------------------------------\n")

    # Compare GPT-4 prediction with actual label
    if gpt_label is not None:
        total_predictions += 1
        if gpt_label == true_label:
            correct_predictions += 1

# Calculate accuracy
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
    print(f"GPT-4 Accuracy on SST-2 test set (for {total_predictions} samples): {accuracy * 100:.2f}%")
else:
    print("No valid predictions made by GPT-4.")

