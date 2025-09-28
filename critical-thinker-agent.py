from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load env variables for os to see
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

print(gemini_api_key)

# Create llm object - automatically loads the GOGGLE_API_KEY from .env file
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro"
)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert critical thinker who understands details very well. You can identify conflicting statements/information in given text"),
        ("human", "Analyze this text: '{text_to_analyze}'")
    ]
)

chain = template | llm

response = chain.invoke(
    {"text_to_analyze": "It rained the whole day today. Since I was getting bored, I got my keys to the car and went to see my friend while listening to the music on this sunny day."}
)

print(response.content)