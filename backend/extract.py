import pypdf
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("Extracting text from PDF...")
reader = pypdf.PdfReader('data/FeePal Report.pdf')
text = ''.join(p.extract_text() for p in reader.pages)

print(f"Extracted {len(text)} characters. Asking Gemini to summarize into a knowledge base...")
model = genai.GenerativeModel('gemini-2.5-flash')
prompt = "You are an expert data extractor. Here is a large document. Extract all the key information, features, business logic, technical details, and any other relevant information about the 'FeePal App' into a comprehensive, highly-detailed text document. Keep as much detail as possible. Document:\n\n" + text

response = model.generate_content(prompt)

print("Writing to feepal.txt...")
with open('data/feepal.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)

print("Cleaning up old PDF...")
os.remove('data/FeePal Report.pdf')

print("Extraction complete!")
