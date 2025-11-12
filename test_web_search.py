
import requests
import newspaper

def test_url_accessibility(url):
    # Test 1: Basic HTTP request
    try:
        response = requests.head(url, timeout=10)
        print(f"HTTP Status: {response.status_code}")
        if response.status_code == 403:
            print("❌ BLOCKED: 403 Forbidden (anti-bot protection)")
        elif response.status_code == 429:
            print("❌ BLOCKED: 429 Too Many Requests (rate limited)")
        elif response.status_code == 200:
            print("✅ OK: Basic access allowed")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    # Test 2: Newspaper3k extraction
    try:
        article = newspaper.Article(url)
        article.download()
        article.parse()
        if article.title and article.text:
            print("✅ OK: Content successfully extracted")
            print(f"Title: {article.title[:]}...")
        else:
            print("❌ FAILED: No content extracted")
    except Exception as e:
        print(f"❌ FAILED: Newspaper3k error - {e}")

# Test the Hindustan Times URL
test_url_accessibility("https://www.thehindu.com/news/national/uttar-pradesh/high-security-alert-in-uttar-pradeshs-bareilly-division-drones-deployed/article70116983.ece")
