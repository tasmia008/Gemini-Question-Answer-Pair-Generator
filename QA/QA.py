import os
import time
from dotenv import load_dotenv
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure Genai Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Google Gemini Model and provide queries as response
def get_gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro', generation_config={
        "temperature": 0,
    })
    response = model.generate_content([prompt])
    return response.text

# Read CSV data
df = pd.read_csv('full_and_final.csv')
df = df[15899:]

output_dir = 'Questions-Answers-Pair-Generations-using-GenAI/QA/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each row in DataFrame
for index, row in tqdm(df.iterrows(), total=len(df)):
    desc = row['Description']
    head = row['Heading']

    # # Construct prompt for Gemini model
    # prompt = f"""
    # You are an expert in making questions and answers from given text description.
    # Now your goal is to make at least 3 questions and corresponding answers from the given Bangla text.
    # Keep the relation with {head} during generate question and answer. Always try to generate the answer with more context. 
    # Make sure not to lose any important information. 

    # Bangla text: {desc}
    # """
    
    prompt = f"""
    You are skilled in crafting questions and answers based on a provided text description.
    Your task is to generate at least 3 questions and corresponding answers from the following Bengali text.
    Maintain a strong connection with '{head}' while generating the questions and answers, keep the '{head}' key information with the question also ensure Bengali answers provide more context.
    Make sure not to lose any important information.

    Bangla text: {desc}
    """

    
    # Get response from Gemini model
    response = get_gemini_response(prompt)
    
    # Define file path for output
    file_path = os.path.join(output_dir, f'response_{index}.txt')
    
    # Write response to file
    with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(response)
    
    # time.sleep(.5)  # Introduce a 0.5-second delay
