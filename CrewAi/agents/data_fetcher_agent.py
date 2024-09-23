from datasets import load_dataset
from crewai import CrewAgent

class DataFetcherAgent(CrewAgent):
    def setup(self):
        self.sst2_dataset = load_dataset("glue", "sst2")  # Load SST-2 dataset
        self.num_examples = 50  # Limit to 50 examples for testing

        # Send the data and generated questions to GPT-4 agent
        self.send_to_agent("GPT4QueryAgent", {"task": "query", "data": self.prepare_questions()})

    def prepare_questions(self):
        questions = []
        for example in self.sst2_dataset['test'][:self.num_examples]:
            sentence = example['sentence']
            true_label = example['label']
            question = f"What is the sentiment of the following sentence: '{sentence}'?"
            questions.append({"sentence": sentence, "true_label": true_label, "question": question})
        return questions

if __name__ == "__main__":
    DataFetcherAgent().start()
