"""
Source Reliability Checker Module

This module maintains a database of news source credibility and provides
scoring for source reliability.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from urllib.parse import urlparse
import os
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SourceCredibility:
    """Represents credibility information for a news source."""
    domain: str
    name: str
    credibility_score: float  # 0.0 to 1.0
    category: str  # 'reliable', 'unreliable', 'satire', 'unknown'
    country: Optional[str] = None
    language: Optional[str] = None
    bias_rating: Optional[str] = None  # 'left', 'center', 'right', 'unknown'
    fact_check_status: Optional[str] = None  # 'verified', 'unverified', 'disputed'
    last_updated: Optional[str] = None

@dataclass
class SourceAnalysis:
    """Represents analysis results for a source."""
    domain: str
    credibility_score: float
    category: str
    confidence: float
    factors: Dict[str, float]
    warnings: List[str]

class SourceReliabilityChecker:
    """Handles source credibility checking and scoring."""
    
    def __init__(self, credibility_db_path: str = "data/source_credibility.json"):
        """Initialize the source reliability checker."""
        self.credibility_db_path = credibility_db_path
        
        # Known reliable sources
        self.reliable_sources = {
            # Original US/UK Core
            'bbc.com', 'reuters.com', 'ap.org', 'npr.org', 'cnn.com', 'nytimes.com',
            'washingtonpost.com', 'wsj.com', 'bloomberg.com', 'ft.com', 'economist.com',
            'theguardian.com', 'independent.co.uk', 'telegraph.co.uk', 'abcnews.go.com',
            'cbsnews.com', 'nbcnews.com', 'foxnews.com', 'usatoday.com', 'latimes.com',
            'chicagotribune.com', 'bostonglobe.com', 'miamiherald.com', 'denverpost.com',
            'seattletimes.com', 'dallasnews.com', 'houstonchronicle.com', 'azcentral.com',
            'inquirer.com', 'baltimoresun.com', 'startribune.com', 'philly.com', 'ajc.com',
            'tennessean.com', 'tampabay.com', 'suntimes.com',

            # --- NEW INTERNATIONAL ADDITIONS ---

            # Canada
            'cbc.ca',               # Canadian Broadcasting Corporation
            'theglobeandmail.com',  # The Globe and Mail
            'ctvnews.ca',           # CTV News

            # Australia / New Zealand
            'abc.net.au',           # Australian Broadcasting Corporation
            'smh.com.au',           # The Sydney Morning Herald
            'theage.com.au',        # The Age
            'stuff.co.nz',          # Stuff (New Zealand)

            # Europe
            'dw.com',               # Deutsche Welle (Germany)
            'france24.com',         # France 24 (France)
            'lemonde.fr',           # Le Monde (France)
            'spiegel.de',           # Der Spiegel (Germany)
            'elpais.com',           # El País (Spain)

            # Asia
            'aljazeera.com',        # Al Jazeera (Qatar-based)
            'japantimes.co.jp',     # The Japan Times (Japan)
            'nhk.or.jp',            # NHK (Japan)
            'timesofindia.indiatimes.com', # The Times of India (India)
            'thehindu.com',         # The Hindu (India)
            'ndtv.com',             # NDTV (India)
            'scmp.com',             # South China Morning Post (Hong Kong)
            'channelnewsasia.com',  # CNA (Singapore)

            # Other Global News Agencies
            'afp.com',              # Agence France-Presse (AFP)
        }
        
        # Known unreliable/satire sources
        self.unreliable_sources = {
          # Original Satire
          'theonion.com', 'clickhole.com', 'duffelblog.com', 'babylonbee.com',
          
          # Original Conspiracy / Disinformation
          'infowars.com', 'breitbart.com', 'naturalnews.com', 'prisonplanet.com',
          'worldnewsdailyreport.com', 'beforeitsnews.com', 'davidicke.com',
          'humansarefree.com', 'yournewswire.com', 'thetruthseeker.co.uk',
          'collective-evolution.com', 'rense.com', 'wakingtimes.com', 'disclose.tv',
          'henrymakow.com', 'whatreallyhappened.com',

          # --- NEW INTERNATIONAL & OTHER ADDITIONS ---
          
          # International State-Controlled Media (often considered propaganda)
          'rt.com',               # Russia Today (Russia)
          'sputniknews.com',      # Sputnik News (Russia)
          'globaltimes.cn',       # Global Times (China)
          'xinhuanet.com',        # Xinhua News Agency (China)
          'presstv.ir',           # Press TV (Iran)

          # Other Highly Biased / Conspiracy
          'zerohedge.com',
          'oann.com',             # One America News Network
          'dailywire.com',
          'thegatewaypundit.com',
          'judicialwatch.org',
          
          # International Satire
          'thedailymash.co.uk',   # The Daily Mash (UK)
          'thebeaverton.com',     # The Beaverton (Canada)
          ' Waterfordwhispersnews.com', # Waterford Whispers News (Ireland)
      }
        self.credibility_db = self._load_credibility_database()
        logger.info("Source reliability checker initialized")
    
    def _load_credibility_database(self) -> Dict[str, SourceCredibility]:
        """Load the credibility database from file."""
        try:
            if os.path.exists(self.credibility_db_path):
                with open(self.credibility_db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        domain: SourceCredibility(**info) 
                        for domain, info in data.items()
                    }
            else:
                # Create initial database
                self._create_initial_database()
                return {}
        except Exception as e:
            logger.error(f"Error loading credibility database: {e}")
            return {}
    
    def _create_initial_database(self):
        """Create initial credibility database with known sources."""
        initial_sources = {}
        
        # Add reliable sources
        for domain in self.reliable_sources:
            initial_sources[domain] = {
                'domain': domain,
                'name': self._extract_name_from_domain(domain),
                'credibility_score': 0.8,
                'category': 'reliable',
                'country': 'US',
                'language': 'en',
                'bias_rating': 'center',
                'fact_check_status': 'verified',
                'last_updated': datetime.now().isoformat()
            }
        
        # Add unreliable sources
        for domain in self.unreliable_sources:
            initial_sources[domain] = {
                'domain': domain,
                'name': self._extract_name_from_domain(domain),
                'credibility_score': 0.2,
                'category': 'unreliable',
                'country': 'US',
                'language': 'en',
                'bias_rating': 'unknown',
                'fact_check_status': 'disputed',
                'last_updated': datetime.now().isoformat()
            }
        
        # Save to file
        os.makedirs(os.path.dirname(self.credibility_db_path), exist_ok=True)
        with open(self.credibility_db_path, 'w', encoding='utf-8') as f:
            json.dump(initial_sources, f, indent=2, ensure_ascii=False)
    
    def _extract_name_from_domain(self, domain: str) -> str:
        """Extract a readable name from domain."""
        # Remove common TLDs and subdomains
        name = domain.replace('www.', '').replace('.com', '').replace('.org', '').replace('.net', '')
        return name.title()
    
    def analyze_source(self, url: str) -> SourceAnalysis:
        """Analyze the credibility of a news source."""
        domain = self._extract_domain(url)
        print(f"DEBUG: Analyzing source for domain: {domain}", flush=True)
        logger.info(f"Analyzing source: {domain}")
        
        # Check if we have cached information
        if domain in self.credibility_db:
            cached_info = self.credibility_db[domain]
            print(f"DEBUG: Found in cache with score {cached_info.credibility_score}", flush=True)
            return SourceAnalysis(
                domain=domain,
                credibility_score=cached_info.credibility_score,
                category=cached_info.category,
                confidence=0.9,
                factors={'cached_data': 1.0},
                warnings=[]
            )
        
        # Perform real-time analysis
        return self._analyze_unknown_source(domain)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return url.lower()
    
    def _analyze_unknown_source(self, domain: str) -> SourceAnalysis:
        """Analyze an unknown source using various heuristics."""
        factors = {}
        warnings = []
        score = 0.5  # Start with neutral score
        
        # Check domain characteristics
        print("DEBUG: Running heuristic scoring", flush=True)
        domain_score = self._analyze_domain_characteristics(domain)
        print(f"DEBUG: Domain analysis score: {domain_score}", flush=True)
        factors['domain_analysis'] = domain_score
        score += domain_score * 0.3
        
        # Check for known patterns
        pattern_score = self._check_known_patterns(domain)
        print(f"DEBUG: Pattern analysis score: {pattern_score}", flush=True)
        factors['pattern_analysis'] = pattern_score
        score += pattern_score * 0.2
        
        # Check SSL certificate
        ssl_score = self._check_ssl_certificate(domain)
        
        print(f"DEBUG: SSL analysis score: {ssl_score}", flush=True)
        factors['ssl_analysis'] = ssl_score
        score += ssl_score * 0.1
        
        # Check social media presence
        social_score = self._check_social_media_presence(domain)
        print(f"DEBUG: Social media analysis score: {social_score}", flush=True)
        factors['social_analysis'] = social_score
        score += social_score * 0.1
        
        # Check content quality indicators
        content_score = self._check_content_quality_indicators(domain)
        
        factors['content_analysis'] = content_score
        score += content_score * 0.3
        
        print(f"DEBUG: Final computed score before clamping: {score}", flush=True)
        # Determine category
        if score >= 0.7:
            category = 'reliable'
        elif score <= 0.3:
            category = 'unreliable'
        else:
            category = 'unknown'
        
        # Add warnings for low scores
        if score < 0.4:
            warnings.append("Low credibility score detected")
        if 'unreliable' in domain.lower() or 'fake' in domain.lower():
            warnings.append("Domain name suggests unreliable content")
        
        return SourceAnalysis(
            domain=domain,
            credibility_score=max(0.0, min(1.0, score)),
            category=category,
            confidence=0.6,  # Lower confidence for unknown sources
            factors=factors,
            warnings=warnings
        )
    
    def _analyze_domain_characteristics(self, domain: str) -> float:
        """Analyze domain characteristics for credibility indicators."""
        score = 0.0
        
        # Check domain length (shorter domains often more credible)
        if len(domain) < 15:
            score += 0.2
        elif len(domain) > 30:
            score -= 0.1
        
        # Check for suspicious patterns
        suspicious_patterns = ['news', 'truth', 'real', 'actual', 'fact', 'report']
        if any(pattern in domain.lower() for pattern in suspicious_patterns):
            score -= 0.2
        
        # Check TLD
        credible_tlds = ['.com', '.org', '.edu', '.gov']
        if any(domain.endswith(tld) for tld in credible_tlds):
            score += 0.1
        
        return score
    
    def _check_known_patterns(self, domain: str) -> float:
        """Check for known patterns in domain names."""
        score = 0.0
        
        # Check against known reliable sources
        if domain in self.reliable_sources:
            return 0.8
        
        # Check against known unreliable sources
        if domain in self.unreliable_sources:
            return -0.8
        
        # Check for news-related domains
        news_indicators = ['news', 'times', 'post', 'journal', 'tribune', 'herald']
        if any(indicator in domain.lower() for indicator in news_indicators):
            score += 0.1
        
        return score
    
    def _check_ssl_certificate(self, domain: str) -> float:
        """Check SSL certificate validity (simplified)."""
        try:
            import ssl
            import socket
            
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    return 0.1 if cert else 0.0
        except:
            return 0.0
    
    def _check_social_media_presence(self, domain: str) -> float:
        """Check social media presence by scraping the website for social media links."""
        try:
            # Fetch the homepage
            response = requests.get(f"https://{domain}", timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code != 200:
                return 0.0

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for social media links
            social_platforms = ['twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com', 'youtube.com']
            social_links = soup.find_all('a', href=re.compile(r'https?://(www\.)?(' + '|'.join(re.escape(p) for p in social_platforms) + r')'))

            if social_links:
                return 0.1  # Positive score if social media links found
            else:
                return 0.0

        except Exception as e:
            logger.debug(f"Error checking social media presence for {domain}: {e}")
            return 0.0
        return 0.0
    
    def _check_content_quality_indicators(self, domain: str) -> float:
        """Check content quality indicators."""
        # This would ideally analyze actual content
        # For now, return neutral score
        return 0.0
    
    def update_source_credibility(self, domain: str, credibility_info: SourceCredibility):
        """Update credibility information for a source."""
        self.credibility_db[domain] = credibility_info
        self._save_credibility_database()
    
    def _save_credibility_database(self):
        """Save the credibility database to file."""
        try:
            os.makedirs(os.path.dirname(self.credibility_db_path), exist_ok=True)
            data = {
                domain: asdict(info) 
                for domain, info in self.credibility_db.items()
            }
            with open(self.credibility_db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving credibility database: {e}")
    
    def get_source_statistics(self) -> Dict:
        """Get statistics about the credibility database."""
        total_sources = len(self.credibility_db)
        reliable_sources = sum(1 for info in self.credibility_db.values() if info.category == 'reliable')
        unreliable_sources = sum(1 for info in self.credibility_db.values() if info.category == 'unreliable')
        unknown_sources = sum(1 for info in self.credibility_db.values() if info.category == 'unknown')
        
        return {
            'total_sources': total_sources,
            'reliable_sources': reliable_sources,
            'unreliable_sources': unreliable_sources,
            'unknown_sources': unknown_sources,
            'avg_credibility_score': sum(info.credibility_score for info in self.credibility_db.values()) / total_sources if total_sources > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    checker = SourceReliabilityChecker()
    
    # Test with various sources
    test_urls = [
        "https://www.bbc.com/news",
        "https://www.reuters.com",
        "https://www.theonion.com",
        "https://www.infowars.com",
        "https://economictimes.indiatimes.com"
    ]
    
    for url in test_urls:
        analysis = checker.analyze_source(url)
        print(f"\nSource: {url}")
        print(f"Domain: {analysis.domain}")
        print(f"Credibility Score: {analysis.credibility_score:.2f}")
        print(f"Category: {analysis.category}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Factors: {analysis.factors}")
        if analysis.warnings:
            print(f"Warnings: {', '.join(analysis.warnings)}")
    
    # Print statistics
    stats = checker.get_source_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total Sources: {stats['total_sources']}")
    print(f"Reliable: {stats['reliable_sources']}")
    print(f"Unreliable: {stats['unreliable_sources']}")
    print(f"Unknown: {stats['unknown_sources']}")
    print(f"Average Credibility: {stats['avg_credibility_score']:.2f}")
