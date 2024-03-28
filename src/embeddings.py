import os
from openai import OpenAI

# Initialize the OpenAI client with the API key from an environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=api_key)

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Fetches the embedding for a single text string using the specified OpenAI model.
    Newlines are replaced with spaces to avoid formatting issues.

    Parameters:
    - text: The text string to get the embedding for.
    - model: The OpenAI model to use for generating the embedding. Defaults to "text-embedding-3-small".

    Returns:
    The embedding as a list of floats.
    """
    # Replace newlines with spaces to ensure the text is on a single line
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
if __name__ == "__main__":
    text = "This is a test."
    embedding = get_embedding(text)
    print(embedding)
