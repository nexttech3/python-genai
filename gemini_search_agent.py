"""A Gemini agent that uses Google Search and function calling to answer user queries."""
"""A Gemini agent that retrieves and analyzes full webpage content for a given query or brand, then generates a consolidated report using Google Search and text extraction capabilities."""
import google.generativeai as genai
import os
import json
import typing
from google.generativeai.types import Tool, GoogleSearchRetrieval
import requests
from bs4 import BeautifulSoup
from newspaper import Article # For newspaper3k

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

def fetch_and_clean_webpage_content(url: str) -> typing.Optional[dict[str, str]]:
    """
    IMPORTANT WEB SCRAPING CONSIDERATIONS:
    - Always respect a website's robots.txt and Terms of Service.
    - Be mindful of the frequency of your requests to avoid overloading servers.
    - Web page structures change frequently; this function's selectors might break.
    - For large-scale or critical applications, consider dedicated news/web APIs.
    - This function uses a generic User-Agent; some sites may still block it.

    Fetches and extracts the main content of a webpage.

    Uses newspaper3k for robust extraction. Falls back to requests/BeautifulSoup
    for basic HTML text if newspaper3k fails or is too complex for the environment.

    Args:
        url: The URL of the webpage to process.

    Returns:
        A dictionary with "url", "title", and "main_content" if successful,
        otherwise None.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Attempt with newspaper3k first
        try:
            article = Article(url)
            article.download()
            article.parse()
            title = article.title if article.title else url # Fallback title
            main_content = article.text
            if main_content: # Ensure content was extracted
                print(f"Successfully extracted content from {url} using newspaper3k.")
                return {"url": url, "title": title, "main_content": main_content}
            else:
                print(f"Newspaper3k extracted no main content from {url}. Attempting fallback.")
        except Exception as e_np:
            print(f"Newspaper3k failed for {url}: {e_np}. Attempting fallback with requests/BeautifulSoup.")

        # Fallback to requests and BeautifulSoup
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors

        if 'text/html' not in response.headers.get('Content-Type', ''):
            print(f"Skipping non-HTML content at {url}")
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # Basic text extraction (can be improved with more specific selectors)
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        main_content = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Try to get a title
        title_tag = soup.find('title')
        title = title_tag.string.strip() if title_tag else url # Fallback title

        if main_content:
            print(f"Successfully extracted content from {url} using requests/BeautifulSoup.")
            return {"url": url, "title": title, "main_content": main_content}
        else:
            print(f"Fallback requests/BeautifulSoup extracted no main content from {url}.")
            return None

    except requests.exceptions.RequestException as e_req:
        print(f"Requests error while fetching {url}: {e_req}")
        return None
    except Exception as e_general:
        print(f"An unexpected error occurred while processing {url}: {e_general}")
        return None

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
  MAX_URLS_TO_PROCESS = 3
  google_search_tool = Tool(google_search_retrieval=GoogleSearchRetrieval())
  # Step 1: Get relevant web pages
  tools_list_step1 = [google_search_tool]
  config_step1 = genai.types.GenerateContentConfig(tools=tools_list_step1)

  instructional_prompt_step1 = f"""User query: {query}

You have access to Google Search. Your task is to find relevant web pages for the user's query.
Please return a list of these pages, including their titles and URLs.
Format your response as a JSON object containing a single key "results", which is a list of dictionaries, where each dictionary has a "title" and a "url".

For example:
{{"results": [{{"title": "Example Page Title", "url": "http://example.com/page1"}}]}}

Do NOT use the 'process_search_results' tool for this step. Only use Google Search.
"""

  try:
    print("Step 1: Searching for relevant web pages...")
    response_step1 = model.generate_content(instructional_prompt_step1, generation_config=config_step1)
    all_page_analyses = []

    if hasattr(response_step1, 'text'):
      try:
        print("Attempting to parse search results (URLs and Titles) JSON...")
        search_results_json = json.loads(response_step1.text)
        if "results" in search_results_json and isinstance(search_results_json["results"], list):
          extracted_items = search_results_json["results"]
          print(f"Found {len(extracted_items)} URLs and Titles. Processing up to {MAX_URLS_TO_PROCESS} of them.")

          for i, item in enumerate(extracted_items):
            if i >= MAX_URLS_TO_PROCESS:
              print(f"Reached MAX_URLS_TO_PROCESS ({MAX_URLS_TO_PROCESS}). Stopping URL processing.")
              break
            
            url = item.get("url")
            title = item.get("title", "N/A")

            if not url:
              print(f"Skipping item {i+1} due to missing URL.")
              continue

            print(f"\nProcessing URL {i+1}/{len(extracted_items)} (max {MAX_URLS_TO_PROCESS}): {url}")
            fetched_data = fetch_and_clean_webpage_content(url)

            if fetched_data and fetched_data.get('main_content'):
              cleaned_text = fetched_data['main_content']
              brand_name = query # Assuming the initial query is the brand name
              
              analysis_prompt = f"""
Context: Article about '{brand_name}' from URL: {url}
Full text:
---
{cleaned_text}
---

Based *only* on the full text provided above, please extract the following information regarding '{brand_name}' and structure your response as a single JSON object with the keys "pontos_principais", "entidades_mencionadas", and "citacao_relevante":

1.  "pontos_principais": A list of 3-5 key discussion points about '{brand_name}'.
2.  "entidades_mencionadas": A list of other people, organizations, or locations mentioned in conjunction with '{brand_name}'.
3.  "citacao_relevante": A single, brief, and directly relevant quote concerning '{brand_name}', if one exists. If not, use an empty string.

Example JSON output format:
{{
    "url": "{url}",
    "title": "{title}",
    "pontos_principais": ["Ponto 1 sobre a marca.", "Ponto 2.", "Ponto 3."],
    "entidades_mencionadas": ["Entidade A", "Organização B"],
    "citacao_relevante": "Citação textual aqui."
}}
"""
              print(f"Step 2: Analyzing content from {url}...")
              analysis_response = model.generate_content(analysis_prompt) # No tools for this call

              if hasattr(analysis_response, 'text'):
                try:
                  print(f"Attempting to parse analysis JSON for {url}...")
                  # It's critical that Gemini actually returns a string that IS JSON.
                  # The prompt asks for it, but we need to be careful here.
                  # Sometimes the model might return ```json\n{...}\n```
                  raw_text = analysis_response.text
                  if raw_text.strip().startswith("```json"):
                      raw_text = raw_text.strip()[7:-3].strip() # Remove markdown
                  
                  page_analysis = json.loads(raw_text)
                  # Ensure URL and title from the original item are part of the analysis for context
                  page_analysis["url"] = url 
                  page_analysis["title"] = title
                  all_page_analyses.append(page_analysis)
                  print(f"Successfully parsed analysis for {url}.")
                except json.JSONDecodeError as json_e:
                  print(f"Error: Failed to parse JSON from analysis response for {url}. Error: {json_e}")
                  print("Raw analysis response text:", analysis_response.text)
              else:
                print(f"Error: Analysis response object for {url} does not have a 'text' attribute.")
                print("Full analysis response:", analysis_response)
            else:
              print(f"Skipping analysis for {url} as content fetching failed or content was empty.")
        else:
          print("Error: JSON response for search results does not contain 'results' list or is malformed.")
          print("Raw search results response text:", response_step1.text)
      except json.JSONDecodeError:
        print("Error: Failed to parse JSON from initial search results response.")
        print("Raw search results response text:", response_step1.text)
    else:
      print("Error: Initial search response object does not have a 'text' attribute.")
      print("Full initial search response:", response_step1)

    # Step 3: Consolidate analyses and generate final report
    if not all_page_analyses:
        print(f"No pages could be analyzed for '{query}'. Unable to generate a report.")
        return

    collected_json_analyses = json.dumps(all_page_analyses, indent=2, ensure_ascii=False)
    count_analyzed_pages = len(all_page_analyses)
    brand_name = query # Assuming the initial query is the brand name

    final_report_prompt = f"""
Você analisou {count_analyzed_pages} páginas web sobre '{brand_name}'. Com base nas seguintes análises individuais em formato JSON:
{collected_json_analyses}

Por favor, gere um relatório consolidado em TEXTO SIMPLES (não JSON) que cubra:
1. Uma contagem total de menções (baseado no número de páginas analisadas com sucesso).
2. Um resumo dos temas principais discutidos sobre '{brand_name}' em todas as fontes.
3. Uma lista consolidada de entidades frequentemente mencionadas junto com '{brand_name}'.
4. Destaque quaisquer citações particularmente relevantes encontradas.

O relatório deve ser claro, conciso e bem organizado.
"""
    print("\nGenerating final report...")
    final_response = model.generate_content(final_report_prompt) # No tools for this call
    
    if hasattr(final_response, 'text'):
        print("\n=== Relatório Final ===")
        print(final_response.text)
    else:
        print("\nCould not generate final report. Raw final response:")
        print(final_response)

  except Exception as e:
    print(f"An error occurred during web search and analysis: {e}")

if __name__ == "__main__":
  # This block demonstrates how to use the run_web_search_agent function.
  # It will use the configured Gemini model and tools to try and answer the query.
  
  # Example usage for brand/topic monitoring:
  # Replace "OpenAI Sora" with the brand/term you want to monitor.
  brand_to_monitor = "OpenAI Sora"
  print(f"Attempting to generate a report for mentions of: '{brand_to_monitor}'")
  run_web_search_agent(brand_to_monitor)

  print("\n" + "="*50 + "\n") # Separator
  
  # Example with a more specific query that might return different types of articles
  specific_query = "latest advancements in renewable energy storage"
  print(f"Attempting to generate a report for: '{specific_query}'")
  run_web_search_agent(specific_query)
