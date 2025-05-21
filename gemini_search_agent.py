"""A Gemini agent that uses Google Search and function calling to answer user queries."""
import google.generativeai as genai
import os
import json
import typing
from google.generativeai.types import Tool, GoogleSearchRetrieval

# IMPORTANT: Set your GEMINI_API_KEY environment variable for this script to work.
# For testing, you can temporarily hardcode it like:
# api_key = "YOUR_API_KEY_HERE" 
# and then use genai.configure(api_key=api_key)
# REMEMBER TO REMOVE THE HARDCODED KEY BEFORE PRODUCTION OR COMMIT.
api_key_from_env = os.getenv("GEMINI_API_KEY")
if not api_key_from_env:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=api_key_from_env)

# Initialize the generative model
model = genai.GenerativeModel("gemini-1.5-pro-latest")

def process_search_results(search_query: str, search_results_json: str) -> str:
  """Processes raw search results from the Google Search tool.
  This function is intended to be called by the Gemini model.
  """
  try:
    parsed_results = json.loads(search_results_json)
    return f"Processed results for query '{search_query}':\n{json.dumps(parsed_results, indent=2)}"
  except json.JSONDecodeError:
    return f"Could not parse search results for query '{search_query}'. Raw results: {search_results_json}"

def run_web_search_agent(query: str):
  """Orchestrates a web search using Gemini with Google Search and a processing function.
  Takes a user query and API key (though API key is configured globally).
  """
  google_search_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())
  tools_list = [google_search_tool, process_search_results]
  config = genai.types.GenerateContentConfig(tools=tools_list)

  instructional_prompt = f"""User query: {query}

You have two tools available:
1. Google Search: to find information on the web.
2. process_search_results(search_query: str, search_results_json: str): to process and summarize raw JSON search results.

Please follow these steps:
1. Use Google Search to find information relevant to the user's query: "{query}".
2. Take the JSON output from Google Search and call the `process_search_results` function with the original query ("{query}") and the search JSON.
3. Based on the processed results from `process_search_results`, provide a comprehensive answer to the user's query.
"""

  try:
    response = model.generate_content(instructional_prompt, generation_config=config)
    # Assuming the response object has a 'text' attribute or similar
    # Adjust based on the actual structure of the response object
    if hasattr(response, 'text'):
      print(response.text)
    else:
      # If the response structure is different, print the whole response
      # to help with debugging or understanding its structure.
      print(response)
  except Exception as e:
    print(f"An error occurred: {e}")

if __name__ == "__main__":
  # This block demonstrates how to use the run_web_search_agent function.
  # It will use the configured Gemini model and tools to try and answer the query.
  test_query = "What is the weather like in Mountain View, CA?"
  print(f"Searching for: '{test_query}'")
  run_web_search_agent(test_query)

  print("\n" + "="*50 + "\n") # Separator
  test_query_2 = "What are the latest advancements in AI?"
  print(f"Searching for: '{test_query_2}'")
  run_web_search_agent(test_query_2)
