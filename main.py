from langchain.llms import OpenAI
from dotenv import dotenv_values
import os 


# Load the .env Files
config = dotenv_values(".env")


os.environ["OPENAI_API_KEY"]=config["OPENAI_KEY"]
input_Text = input("Enter the text you want to search : ")

## OPENAI LLMS
llm=OpenAI(temperature=0.8)

if input_Text:
    print("the output from the model",llm(input_Text))