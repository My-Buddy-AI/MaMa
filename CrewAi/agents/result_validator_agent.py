from crewai import CrewAgent

class ResultValidatorAgent(CrewAgent):
    def setup(self):
        self.correct_predictions = 0
        self.total_predictions = 0

    def handle_message(self, message):
        task = message['task']
        if task == "validate":
            results = message['data']
            self.compare_results(results)

    def compare_results(self, results):
        for result in results:
            sentence = result["sentence"]
            true_label = result["true_label"]
            gpt_response = result["gpt_response"]

            # Interpreting GPT-4 response
            gpt_label = self.interpret_gpt4_response(gpt_response)

            # Compare GPT-4 prediction with actual label
            if gpt_label is not None:
                self.total_predictions += 1
                if gpt_label == true_label:
                    self.correct_predictions += 1

            # Log the comparison
            self.log(f"Sentence: {sentence}")
            self.log(f"Actual Label: {'Positive' if true_label == 1 else 'Negative'}")
            self.log(f"GPT-4 Response: {gpt_response}")
            self.log(f"GPT-4 Predicted Label: {'Positive' if gpt_label == 1 else 'Negative' if gpt_label == 0 else 'Unclear'}")
            self.log("--------------------------------------------------")

        # Calculate accuracy
        if self.total_predictions > 0:
            accuracy = (self.correct_predictions / self.total_predictions) * 100
            self.log(f"GPT-4 Accuracy: {accuracy:.2f}%")

    def interpret_gpt4_response(self, response):
        if 'positive' in response.lower():
            return 1
        elif 'negative' in response.lower():
            return 0
        else:
            return None  # For unclear responses

if __name__ == "__main__":
    ResultValidatorAgent().start()
