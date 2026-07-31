import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
print(f"loaded a key of length {len(api_key)}")