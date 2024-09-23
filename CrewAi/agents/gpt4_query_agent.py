import openai
from crewai import CrewAgent

class GPT4QueryAgent(CrewAgent):
    def setup(self):
        openai.api_key = 'your-openai-api-key'

    def handle_message(self, message):
        task = message['task']
        if task == "query":
            responses = []
            questions = message['data']
            for question_obj in questions:
                question = question_obj["question"]
                response = self.query_gpt4(question)
                question_obj["gpt_response"] = response
                responses.append(question_obj)

            # Send responses to the result validator agent
            self.send_to_agent("ResultValidatorAgent", {"task": "validate", "data": responses})

    def query_gpt4(self, question):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    GPT4QueryAgent().start()
