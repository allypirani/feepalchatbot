import google.generativeai as genai
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("Reading current feepal.txt...")
with open('data/feepal.txt', 'r', encoding='utf-8') as f:
    current_text = f.read()

print("Asking Gemini to expand the document with the new details...")
model = genai.GenerativeModel('gemini-2.5-flash')
prompt = f"""
Here is the current documentation for the FeePal App. 
The user wants to expand this document significantly. Please rewrite it and make it much more detailed.

Crucial Requirements to add:
1. Explicitly state that FeePal is a Flutter-based Mobile Application.
2. Explicitly state that it uses Firebase as the backend and Firestore as the database.
3. Flesh out and add more relevant information about its potential architecture, technical stack, advanced features, and use cases.
4. Add a new section discussing alternatives to the FeePal app (e.g., existing school fee management systems).

Keep all the original information and features completely intact, just weave the new details in and make it a highly comprehensive text document.

Original Document:
{current_text}
"""

response = model.generate_content(prompt)

print("Writing expanded document back to feepal.txt...")
with open('data/feepal.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Deleting old ChromaDB cache so the chatbot re-ingests the new file...")
if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")

print("Expansion complete!")
