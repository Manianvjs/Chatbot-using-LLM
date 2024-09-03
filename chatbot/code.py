import openai
import os
import pandas as pd
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Make sure to set your OpenAI API key in an environment variable or replace `os.getenv('OPENAI_API_KEY')` with your API key directly
openai.api_key = os.getenv('OPENAI_API_KEY')

class DatasetChatbot:
    def __init__(self, dataset_path):
        self.dataset = pd.read_csv(dataset_path)
        self.qa_pairs = list(zip(self.dataset['question'], self.dataset['answer']))
        self.conversation_history = []

    def add_to_history(self, role, content):
        logging.info(f"Adding to history: {role} - {content}")
        self.conversation_history.append({"role": role, "content": content})

    def find_best_answer(self, user_input):
        logging.info(f"Finding best answer for: {user_input}")
        for question, answer in self.qa_pairs:
            if re.search(re.escape(question), user_input, re.IGNORECASE):
                return answer
        return None

    def get_model_response(self, user_input):
        logging.info(f"Getting model response for: {user_input}")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # You can change the model based on availability and needs
                messages=self.conversation_history + [{"role": "user", "content": user_input}]
            )
            assistant_reply = response.choices[0].message["content"]
            return assistant_reply
        except Exception as e:
            logging.error(f"Error getting model response: {e}")
            return "Sorry, there was an error processing your request."

    def run(self):
        print("Chatbot is ready to talk! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Chatbot: Goodbye!")
                break

            # Add user input to history
            self.add_to_history("user", user_input)
            
            # Try to find an answer from the dataset
            response = self.find_best_answer(user_input)

            if response is None:
                # Fallback to model response
                response = self.get_model_response(user_input)

            # Add model response to history
            self.add_to_history("assistant", response)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    dataset_path = 'dataset.csv'  # Path to your dataset file
    chatbot = DatasetChatbot(dataset_path)
    chatbot.run()
