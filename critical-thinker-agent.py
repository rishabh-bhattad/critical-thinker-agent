from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load env variables for os to see
load_dotenv()


# Create llm object - automatically loads the GOGGLE_API_KEY from .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro"
)

# Set the template for the Critical thinker
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert critical thinker who understands details very well. You can identify conflicting statements/information in given text. First tell if the sentence is 'Conflicting' or 'Not Conflicting' and then provide the area where it is conflicting if it is and give reason in less than 1 line."),
        ("human", "Analyze this text: '{text_to_analyze}'")
    ]
)

# Chaining template to llm
chain = template | llm

# Invoking llm from user nput
def run_agentic_analysis(user_text: str):
    return chain.invoke(
        {"text_to_analyze": user_text}
    )

# Main block
if __name__ == "__main__":
    print("Welcome to the critical Thinker Agent! Type 'exit' instead of prompt to quit.")

    # Hard coding it to run only 3 loops before it stops to not utilize resources
    for i in range(3):
        user_input = input("Please enter the text you want to analyze: ")
        if user_input == 'exit':
            break
        response = run_agentic_analysis(user_text=user_input)
        print(response.content)
