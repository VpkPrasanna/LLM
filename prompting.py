import os 
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
# Load the .env Files
config = dotenv_values(".env")


os.environ["OPENAI_API_KEY"]=config["OPENAI_KEY"]

os.environ["OPENAI_API_KEY"]=openai_key
