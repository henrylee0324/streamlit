import os
from dotenv import load_dotenv
from openai import OpenAI
import openai

# Load environment variables
load_dotenv()

api_key = os.getenv("GRAPHRAG_API_KEY")
llm_model = os.getenv("GRAPHRAG_LLM_MODEL")

def compare_responses(question, response1, response2):
    # Define a comprehensive system prompt
    system_prompt = """
    You are an advanced analytical assistant specialized in comparing textual data.
    Your task is to analyze two given responses to a specific question. 
    Using the provided question as context, identify and explain:
    1. How well each response addresses the question.
    2. Key differences between the responses.
    3. Key similarities between the responses.
    4. Specific examples or phrases that highlight these differences and similarities.
    5. A summarized conclusion highlighting which response is more complete or better aligned with the question.
    Provide your analysis in a clear and **structured format** like this:

    Analysis:
    - How well each response addresses the question:
        * Response 1: [Analysis of Response 1]
        * Response 2: [Analysis of Response 2]
    - Key differences:
        1. [Difference 1]
        2. [Difference 2]
    - Key similarities:
        1. [Similarity 1]
        2. [Similarity 2]
    - Examples or phrases:
        * Response 1: "[Example]"
        * Response 2: "[Example]"
    - Conclusion:
        [Your conclusion on which response is better and why]
    """

    # Create the comparison prompt with the question and responses
    prompt = f"""
    Question:
    {question}
    
    Response 1:
    {response1}
    
    Response 2:
    {response2}
    """

    client = OpenAI(api_key=api_key)

    # Send the request to OpenAI
    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return the structured output
    structured_output = completion.choices[0].message.content
    return structured_output

if __name__ == "__main__":
    # Example question and responses
    question = "What are the main causes of climate change?"
    response1 = "Climate change is influenced by greenhouse gases."
    response2 = "Climate change is driven by greenhouse gases and deforestation."

    print("Question:", question)
    print("Response from Dataset A:", response1)
    print("Response from Dataset B:", response2)

    print("\nComparing responses with GPT-4o-mini...")
    differences = compare_responses(question, response1, response2)

    print("\nDifferences identified by GPT-4o-mini:\n", differences)
