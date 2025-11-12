import os

def load_env_file(envfile="env_example.txt"):
    if os.path.exists(envfile):
        print(f"Loading environment file: {envfile}")
        with open(envfile, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value.strip()
        print(f"Loaded environment variables from {envfile}")
    else:
        print(f"Environment file {envfile} not found")

# Load environment variables early
load_env_file("env_example.txt")

# Confirm keys loaded
print("NEWS_API_KEY =", os.getenv("NEWS_API_KEY"))
print("GOOGLE_SEARCH_API_KEY =", os.getenv("GOOGLE_SEARCH_API_KEY"))
print("WORLD_NEWS_API_KEY =", os.getenv("WORLD_NEWS_API_KEY"))

import requests
import base64


def check_ssl(url):
    """Check SSL certificate validity for a URL."""
    try:
        # Use HEAD request to check SSL without downloading content
        response = requests.head(url, timeout=5, verify=True)
        return "Valid SSL"
    except requests.exceptions.SSLError as e:
        return f"Invalid SSL: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"SSL check failed: {str(e)}"
    except Exception as e:
        return f"Error checking SSL: {str(e)}"

# Text input or URL
claim = "ShahRukh Khan becomes world's richest actor with a net worth of $1.5 billion dollars."


payload = {
    "content": claim,
    "content_type": "text",
    "include_multimodal": False  # Enable multimodal verification for images
}

url = "http://127.0.0.1:8000/verify"

print("Sending verification request with payload:")
print(payload)

try:
    response = requests.post(url, json=payload)
    print(f"HTTP Status Code: {response.status_code}")
    result = response.json()
    print("\nResponse received:")

    # Print simple key-value pairs excluding evidence_summary
    for k, v in result.items():
        if k != 'evidence_summary':
            print(f"{k}: {v}")

    # Print evidence summary with details
    print("\nEvidence summary:")
    ev_summary = result.get("evidence_summary", {})
    for k, v in ev_summary.items():
        if k != "used_articles":
            print(f"  {k}: {v}")

    # Print used articles along with their relevance scores
    articles = ev_summary.get("used_articles") or result.get("evidence_articles")
    if articles:
        print("\n--- Sources and Articles Used for Verdict ---")
        for i, article in enumerate(articles):
            title = article.get("title", "N/A")
            source = article.get("source", "N/A")
            url = article.get("url", "N/A")
            snippet = article.get("snippet", "N/A")
            relevance = article.get("relevance_score", "N/A")
            source_reliability = article.get("source_reliability_score", "N/A")
            print(f"\nArticle {i + 1}:")
            print(f"  Title: {title}")
            print(f"  Source: {source}")
            print(f"  URL: {url}")
            print(f"  Snippet: {snippet}")
            print(f"  Relevance Score: {relevance}")
            print(f"  Source Reliability Score: {source_reliability}")
            print(f"  SSL Score: {check_ssl(url)}")
    else:
        print("\nNo article list found in evidence summary.")

    # Print ALL articles retrieved with API source and scores
    all_articles = result.get("all_retrieved_articles") or ev_summary.get("all_articles") or result.get("search_results")
    if all_articles:
        print("\n--- ALL Retrieved Articles (Regardless of Scores) ---")
        for i, article in enumerate(all_articles):
            title = article.get("title", "N/A")
            source = article.get("source", "N/A")
            url = article.get("url", "N/A")
            snippet = article.get("snippet", "N/A")[:200] + "..." if len(article.get("snippet", "")) > 200 else article.get("snippet", "N/A")
            relevance = article.get("relevance_score", "N/A")
            source_reliability = article.get("source_reliability_score", "N/A")
            api_used = article.get("api_source", article.get("search_engine", "Unknown"))
            print(f"\nArticle {i + 1}:")
            print(f"  Title: {title}")
            print(f"  Source: {source}")
            print(f"  URL: {url}")
            print(f"  Snippet: {snippet}")
            print(f"  Relevance Score: {relevance}")
            print(f"  Source Reliability Score: {source_reliability}")
            print(f"  API Used: {api_used}")
    else:
        print("\nNo all articles list found in response.")

except Exception as e:
    print("Error during request or response parsing:")
    print(e)