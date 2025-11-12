"""
Web Search and Evidence Retrieval Module

This module handles searching for evidence related to claims using various search APIs
and retrieving relevant articles and metadata.
"""

import os
import logging
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import worldnewsapi
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from serpapi import GoogleSearch
import json
from sentence_transformers import SentenceTransformer, util
import re

# Optional NLP for noun phrase extraction
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

from src.verification.source_reliability import SourceReliabilityChecker
from src.search.enhanced_query_generator import EnhancedQueryGenerator

import nltk
from nltk.corpus import stopwords

# Download (if needed) at runtime:
try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sanitize API keys in logs
def sanitize_log_message(message: str) -> str:
    """Remove API keys from log messages."""
    import re
    # Replace potential API key patterns with placeholders
    message = re.sub(r'key:\s*[a-zA-Z0-9_-]{20,}', 'key: [REDACTED]', message)
    message = re.sub(r'apiKey:\s*[a-zA-Z0-9_-]{20,}', 'apiKey: [REDACTED]', message)
    message = re.sub(r'access_key:\s*[a-zA-Z0-9_-]{20,}', 'access_key: [REDACTED]', message)
    message = re.sub(r'apikey:\s*[a-zA-Z0-9_-]{20,}', 'apikey: [REDACTED]', message)
    return message

@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str]
    relevance_score: float
    domain: str
    content: Optional[str] = None

@dataclass
class Evidence:
    """Represents evidence found for a claim."""
    claim_text: str
    search_results: List[SearchResult]
    all_retrieved_articles: List[SearchResult]
    total_results: int
    search_queries: List[str]
    timestamp: str

class WebSearchEngine:
    """Handles web search and evidence retrieval for claims."""

    def __init__(self):
        """Initialize search engines and APIs."""
        self.source_checker = SourceReliabilityChecker()
        self.query_generator = EnhancedQueryGenerator()

        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.mediastack_api_key = None
        self.newsdata_api_key = os.getenv('NEWS_DATA_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.bing_api_key = os.getenv('BING_SEARCH_API_KEY')
        self.serpapi_key = None
        self.worldnewsapi_key = os.getenv('WORLD_NEWS_API_KEY')

        # Additional free news APIs
        self.currents_api_key = os.getenv('CURRENTS_API_KEY')
        self.contextual_web_search_api_key = os.getenv('CONTEXTUAL_WEB_SEARCH_API_KEY')
        self.the_guardian_api_key = os.getenv('THE_GUARDIAN_API_KEY')

        # Entity anchoring and quality filters
        self.entity_anchors = set()
        self.topic_tokens = set()
        self.numeric_patterns = set()
        self.high_credibility_domains = {
            'forbes.com', 'bloomberg.com', 'bbc.com', 'reuters.com', 'thehindu.com',
            'indianexpress.com', 'livemint.com', 'hollywoodreporter.com', 'variety.com',
            'factcheck.org', 'snopes.com', 'boomlive.in', 'altnews.in', 'nytimes.com',
            'washingtonpost.com', 'ap.org', 'npr.org', 'cnn.com'
        }
        self.blacklisted_paths = {'/tvshowbiz/', '/opinion/', '/blog/'}

    def _extract_entity_anchors(self, claim_text: str):
        """Extract entity anchors, topic tokens, and numeric patterns from claim."""
        # Normalize text
        normalized = re.sub(r'[^\w\s]', ' ', claim_text.lower()).strip()
        
        # Extract multi-word entities (potential names)
        multi_entities = re.findall(r'\b[a-z]{3,}\s+[a-z]{3,}\b', normalized)
        
        # Extract single words (key terms)
        single_words = re.findall(r'\b[a-z]{4,}\b', normalized)  # Longer words
        
        # Topic terms for wealth/celebrity claims
        topic_patterns = r'\b(richest|wealthiest|net worth|networth|billionaire|millionaire|actor|actress|star|celebrity)\b'
        topics = re.findall(topic_patterns, normalized)
        
        # Numeric patterns
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*billion|bn|million|mn)?\b', normalized)
        currencies = re.findall(r'\$?\s*\d+(?:,\d{3})*(?:\.\d{2})?', claim_text)
        
        # Aliases for common entities (expand as needed)
        aliases = {}
        if 'shah rukh khan' in normalized or 'srk' in normalized:
            aliases.update({'shah rukh khan', 'shahrukh khan', 'srk'})
        
        # Combine
        self.entity_anchors = set(multi_entities + single_words[:10] + list(aliases.keys()))
        self.topic_tokens = set(topics)
        self.numeric_patterns = set(numbers + currencies)
        
        logger.debug(f"Extracted anchors: {self.entity_anchors}, topics: {self.topic_tokens}, numerics: {self.numeric_patterns}")
        
        # Initialize NewsAPI client
        if self.news_api_key:
            self.news_api = NewsApiClient(api_key=self.news_api_key)
            logger.info("NewsAPI client initialized")
        else:
            logger.warning("NEWS_API_KEY not found. NewsAPI search will be disabled.")
            self.news_api = None
        
        # Log available APIs
        available_apis = []
        if self.google_api_key:
            available_apis.append("Google Custom Search")
        if self.news_api_key:
            available_apis.append("NewsAPI")
        if self.mediastack_api_key:
            available_apis.append("MediaStack")
        if self.newsdata_api_key:
            available_apis.append("NewsData")
        if self.bing_api_key:
            available_apis.append("Bing Search")
        if self.serpapi_key:
            available_apis.append("SerpAPI")
        if self.worldnewsapi_key:
            available_apis.append("World news API")
        if self.currents_api_key:
            available_apis.append("Currents API")
        if self.contextual_web_search_api_key:
            available_apis.append("Contextual Web Search")
        if self.the_guardian_api_key:
            available_apis.append("The Guardian API")

        logger.info(f"Available search APIs: {', '.join(available_apis) if available_apis else 'None'}")
        
        # Rate limiting
        self.search_delay = 1.0  # seconds between searches

        # Initialize embedding model for relevance scoring
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded for relevance scoring")

        logger.info("Web search engine initialized")
    
    async def search_for_evidence(self, claim_text: str, keywords: List[str] = None) -> Evidence:
        """Search for evidence related to a claim."""
        try:
            # Extract entity anchors and topic tokens for better filtering
            self._extract_entity_anchors(claim_text)
            self.claim_text = claim_text  # Store for use in scoring methods

            # Generate enhanced search queries
            search_queries = await self._generate_search_queries(claim_text, keywords)

            # DEBUG: Print generated queries for visibility
            try:
                print(f"\nGenerated search queries ({len(search_queries)}):")
                print("-" * 50)
                for i, q in enumerate(search_queries, 1):
                    print(f"{i}. {q}")
                print()
            except Exception as e:
                logger.debug(f"Failed to print generated queries: {e}")
            
            # Perform searches using multiple engines (prioritize Google Search)
            all_results = []
            search_engines_used = []
            
            # Google Custom Search (primary)
            if self.google_api_key:
                try:
                    google_results = await self._search_google(search_queries)
                    all_results.extend(google_results)
                    search_engines_used.append("Google")
                    logger.info(f"Google Search: Found {len(google_results)} results")
                except Exception as e:
                    logger.error(f"Google Search failed: {e}")
            
            # NewsAPI search
            if self.news_api:
                try:
                    news_results = await self._search_news_api(search_queries)
                    all_results.extend(news_results)
                    search_engines_used.append("NewsAPI")
                    logger.info(f"NewsAPI: Found {len(news_results)} results")
                except Exception as e:
                    logger.error(f"NewsAPI search failed: {e}")
            
            # MediaStack API search (free alternative)
            if self.mediastack_api_key:
                try:
                    mediastack_results = await self._search_mediastack(search_queries)
                    all_results.extend(mediastack_results)
                    search_engines_used.append("MediaStack")
                    logger.info(f"MediaStack: Found {len(mediastack_results)} results")
                except Exception as e:
                    logger.error(f"MediaStack search failed: {e}")
            
            # NewsData API search (free alternative)
            if self.newsdata_api_key:
                try:
                    newsdata_results = await self._search_newsdata(search_queries)
                    all_results.extend(newsdata_results)
                    search_engines_used.append("NewsData")
                    logger.info(f"NewsData: Found {len(newsdata_results)} results")
                except Exception as e:
                    logger.error(f"NewsData search failed: {e}")
            
            # Bing Search API (optional)
            if self.bing_api_key:
                try:
                    bing_results = await self._search_bing(search_queries)
                    all_results.extend(bing_results)
                    search_engines_used.append("Bing")
                    logger.info(f"Bing Search: Found {len(bing_results)} results")
                except Exception as e:
                    logger.error(f"Bing search failed: {e}")
            
            # SerpAPI search (if available)
            if self.serpapi_key:
                print(f"DEBUG: Starting SerpAPI search")
                try:
                    serpapi_results = await self._search_serpapi(search_queries)
                    all_results.extend(serpapi_results)
                    search_engines_used.append("SerpAPI")
                    logger.info(f"SerpAPI: Found {len(serpapi_results)} results")
                    print(f"DEBUG: SerpAPI search completed")
                except Exception as e:
                    logger.error(f"SerpAPI search failed: {e}")
                    print(f"DEBUG: SerpAPI search failed with exception: {e}")

            if self.worldnewsapi_key:
                print(f"DEBUG: Starting WorldNewsAPI search")
                try:
                    worldnewsapi_results = await self._search_world_news_api(search_queries)
                    all_results.extend(worldnewsapi_results)
                    search_engines_used.append("WorldnewsAPI")
                    logger.info(f"WorldNewsAPI: Found {len(worldnewsapi_results)} results")
                    print(f"DEBUG: WorldNewsAPI search completed")
                except Exception as e:
                    logger.error(f"WorldnewsAPI search failed: {e}")
                    print(f"DEBUG: WorldNewsAPI search failed with exception: {e}")

            # Currents API search
            if self.currents_api_key:
                print(f"DEBUG: Starting Currents API search")
                try:
                    currents_results = await self._search_currents_api(search_queries)
                    all_results.extend(currents_results)
                    search_engines_used.append("Currents API")
                    logger.info(f"Currents API: Found {len(currents_results)} results")
                    print(f"DEBUG: Currents API search completed")
                except Exception as e:
                    logger.error(f"Currents API search failed: {e}")
                    print(f"DEBUG: Currents API search failed with exception: {e}")

            # Contextual Web Search API search
            if self.contextual_web_search_api_key:
                print(f"DEBUG: Starting Contextual Web Search")
                try:
                    contextual_results = await self._search_contextual_web_search(search_queries)
                    all_results.extend(contextual_results)
                    search_engines_used.append("Contextual Web Search")
                    logger.info(f"Contextual Web Search: Found {len(contextual_results)} results")
                    print(f"DEBUG: Contextual Web Search completed")
                except Exception as e:
                    logger.error(f"Contextual Web Search failed: {e}")
                    print(f"DEBUG: Contextual Web Search failed with exception: {e}")

            # The Guardian API search
            if self.the_guardian_api_key:
                try:
                    guardian_results = await self._search_the_guardian(search_queries)
                    all_results.extend(guardian_results)
                    search_engines_used.append("The Guardian API")
                    logger.info(f"The Guardian API: Found {len(guardian_results)} results")
                except Exception as e:
                    logger.error(f"The Guardian API search failed: {e}")

            logger.info(f"Search completed using: {', '.join(search_engines_used) if search_engines_used else 'No engines'}")

            # Remove duplicates
            unique_results = self._deduplicate_results(all_results)

            # Apply robust pre-filtering pipeline
            filtered_results = self._apply_robust_pre_filters(unique_results, claim_text)

            # Rank filtered results with calibrated scoring
            ranked_results = self._rank_results_with_calibrated_scoring(filtered_results, claim_text)

            # Fetch full content for all ranked results (no cap)
            top_results = await self._fetch_content_for_top_results(ranked_results)

            return Evidence(
                claim_text=claim_text,
                search_results=top_results,
                all_retrieved_articles=all_results,
                total_results=len(ranked_results),
                search_queries=search_queries,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error searching for evidence: {e}")
            return Evidence(
                claim_text=claim_text,
                search_results=[],
                total_results=0,
                search_queries=[],
                timestamp=datetime.now().isoformat()
            )
    
    def _truncate_query(self, query: str, max_length: int = 100) -> str:
        """Truncate a query while maintaining meaning and staying under max_length."""
        if len(query) <= max_length:
            return query
            
        # Remove quotes if present to handle content better
        is_quoted = query.startswith('"') and query.endswith('"')
        if is_quoted:
            query = query[1:-1]
            max_length -= 2  # Account for quotes we'll add back
            
        # Split into words and rebuild query
        words = query.split()
        result = []
        current_length = 0
        
        # Always include first word (usually action/verb) and key entities
        result.append(words[0])
        current_length = len(words[0])
        
        # Try to include key entities and verbs
        key_terms = ["fact", "check", "verify", "debunk", "official", "source"]
        for word in words[1:]:
            # Calculate length with space
            word_length = len(word) + 1  # +1 for space
            
            # Always include key verification terms if possible
            if any(term in word.lower() for term in key_terms):
                if current_length + word_length <= max_length:
                    result.append(word)
                    current_length += word_length
                continue
                
            # For other words, include if space permits
            if current_length + word_length <= max_length:
                result.append(word)
                current_length += word_length
            else:
                break
                
        truncated = " ".join(result)
        if is_quoted:
            truncated = f'"{truncated}"'
            
        return truncated
        
    async def _generate_search_queries(self, claim_text: str, keywords: List[str] = None) -> List[str]:
        """Generate enhanced search queries with entity anchoring and provider constraints."""
        if not claim_text:
            return []

        # Config for query limits
        self.enable_length_limit = True
        self.max_query_length = 100

        # Use EnhancedQueryGenerator for base queries
        query_results = await self.query_generator.generate_queries(
            claim_text,
            languages=['en']
        )

        # Priority queries with entity anchoring
        priority_queries = set()

        # 1. Entity-anchored base queries
        base_claim = self._truncate_query(claim_text.strip()) if self.enable_length_limit else claim_text.strip()
        for anchor in self.entity_anchors:
            if anchor not in base_claim.lower():
                anchored = f"{anchor} {base_claim}"
                if self.enable_length_limit:
                    anchored = self._truncate_query(anchored)
                priority_queries.add(anchored)
                priority_queries.add(f'"{anchor}" {base_claim}')

        # 2. Quoted entity + topic combinations
        for anchor in self.entity_anchors:
            for topic in self.topic_tokens:
                q = f'"{anchor}" {topic}'
                if self.enable_length_limit:
                    q = self._truncate_query(q)
                priority_queries.add(q)

        # 3. Numeric-anchored queries
        for num in self.numeric_patterns:
            for anchor in self.entity_anchors:
                q = f"{anchor} {num}"
                if self.enable_length_limit:
                    q = self._truncate_query(q)
                priority_queries.add(q)

        # 4. Fact-checking with anchors
        fact_check_bases = [base_claim]
        for anchor in self.entity_anchors:
            fact_check_bases.append(anchor)

        fact_check_queries = []
        for base in fact_check_bases[:3]:  # Limit bases
            fc_queries = [
                f"fact check {base}",
                f"verify {base}",
                f"debunk {base}",
                f"{base} hoax",
                f"{base} false"
            ]
            if self.enable_length_limit:
                fc_queries = [self._truncate_query(q) for q in fc_queries]
            fact_check_queries.extend(fc_queries)

        priority_queries.update(fact_check_queries[:5])  # Limit fact-check queries

        # 5. Site-specific queries for high-credibility sources
        for site in self.high_credibility_domains:
            for anchor in list(self.entity_anchors)[:2]:  # Top 2 anchors
                q = f'site:{site} {anchor}'
                if self.enable_length_limit:
                    q = self._truncate_query(q)
                priority_queries.add(q)

        # 6. Temporal constraints (recent 12-24 months)
        recent_queries = []
        for q in priority_queries.copy():
            recent_q = f"{q} after:2022"  # Google-style date filter
            if self.enable_length_limit:
                recent_q = self._truncate_query(recent_q)
            recent_queries.append(recent_q)

        priority_queries.update(recent_queries[:3])  # Limit recent variants

        # 7. Integrate EnhancedQueryGenerator results (filtered for anchors)
        generator_queries = []
        for qr in query_results:
            q = qr["query"]
            # Only include if it contains at least one entity anchor or topic
            q_lower = q.lower()
            if any(anchor in q_lower for anchor in self.entity_anchors) or \
               any(topic in q_lower for topic in self.topic_tokens):
                if self.enable_length_limit:
                    q = self._truncate_query(q)
                generator_queries.append(q)

        priority_queries.update(generator_queries[:5])

        # 8. Keywords integration
        if keywords:
            for keyword in keywords[:3]:
                for anchor in self.entity_anchors:
                    q = f"{anchor} {keyword}"
                    if self.enable_length_limit:
                        q = self._truncate_query(q)
                    priority_queries.add(q)

        # Avoid number-only queries
        final_queries = [q for q in priority_queries if not re.match(r'^\s*\d+\s*(billion|million)?\s*$', q.strip().lower())]

        # Deduplicate and limit
        seen = set()
        ordered = []
        for q in list(priority_queries):
            if q not in seen and len(q.split()) >= 3:  # At least 3 words
                seen.add(q)
                ordered.append(q)
                if len(ordered) >= 15:  # Cap at 15 for efficiency
                    break

        logger.info(f"Generated {len(ordered)} anchored queries (anchors: {len(self.entity_anchors)}, topics: {len(self.topic_tokens)})")

        return ordered[:10]  # Final limit for APIs
    
    def _extract_entities_for_search(self, text: str) -> List[str]:
        """Extract important entities for search queries."""
        # Simple entity extraction - in production, use NER
        import re
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+\b', text)
        
        # Combine and filter
        entities = capitalized_words + numbers
        return [e for e in entities if len(e) > 2][:5]  # Top 5 entities

    
    
    def _extract_domains_from_text(self, text: str) -> List[str]:
        """Extract domains present in the text to bias site: queries dynamically."""
        try:
            domains: List[str] = []
            for m in re.finditer(r'(?:https?://)?([A-Za-z0-9.-]+\.[A-Za-z]{2,})', text):
                domain = m.group(1).lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                if domain not in domains:
                    domains.append(domain)
            return domains
        except Exception:
            return []

    def _svo_phrases_spacy(self, text: str) -> List[str]:
        """Extract SVO phrases using spaCy dependency parsing with quality filters (generic)."""
        phrases: List[str] = []
        doc = _NLP(text)
        for sent in doc.sents:
            verbs = [t for t in sent if t.pos_ == 'VERB']
            for v in verbs:
                # skip auxiliaries and light verbs
                if v.pos_ == 'AUX' or v.lemma_ in {'be', 'have', 'do'}:
                    continue
                subs = [c for c in v.children if c.dep_ in ('nsubj', 'nsubjpass')]
                objs = [c for c in v.children if c.dep_ in ('dobj', 'obj', 'attr')]
                preps = [c for c in v.children if c.dep_ == 'prep']
                for s in subs:
                    # subject pronoun without named/object context -> skip
                    if s.pos_ == 'PRON':
                        obj_has_propn = any(t.pos_ == 'PROPN' for o in objs for t in o.subtree) or any(
                            (gc.dep_ == 'pobj' and gc.pos_ == 'PROPN') for p in preps for gc in p.children
                        )
                        if not obj_has_propn:
                            continue
                    if objs:
                        for o in objs:
                            # build base phrase
                            phrase = f"{s.text} {v.lemma_} {o.text}"
                            # attach prepositional objects
                            extras: List[str] = []
                            for p in preps:
                                pobj = next((gc for gc in p.children if gc.dep_ == 'pobj'), None)
                                if pobj is not None:
                                    extras.append(f"{p.text} {pobj.text}")
                            if extras:
                                phrase = f"{phrase} {' '.join(extras)}"
                            # length and content checks
                            tokens = [t for t in re.findall(r"[A-Za-z0-9'-]+", phrase)]
                            if not (3 <= len(tokens) <= 12):
                                continue
                            # require at least one proper noun in subject/object/extras or sentence
                            if not (any(t.pos_ == 'PROPN' for t in s.subtree) or any(t.pos_ == 'PROPN' for t in o.subtree)):
                                if not any(t.pos_ == 'PROPN' for t in sent):
                                    continue
                            # avoid pronoun-led generic phrases
                            if tokens[0].lower() in {'it', 'its', 'they', 'their', 'this', 'that'}:
                                continue
                            phrases.append(phrase)
                    else:
                        # subject-verb with context span
                        span_tokens = sorted([t for t in s.subtree] + [t for t in v.subtree], key=lambda t: t.i)
                        span = doc[span_tokens[0].i:span_tokens[-1].i+1].text
                        tokens = [t for t in re.findall(r"[A-Za-z0-9'-]+", span)]
                        if not (3 <= len(tokens) <= 12):
                            continue
                        if not any(t.pos_ == 'PROPN' for t in sent):
                            continue
                        if tokens and tokens[0].lower() in {'it', 'its', 'they', 'their', 'this', 'that'}:
                            continue
                        phrases.append(span)
        if not phrases:
            phrases = [s.text.strip() for s in doc.sents if s.text.strip()]
        # dedupe preserving order
        seen = set()
        out = []
        for p in phrases:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _rank_phrases(self, phrases: List[str], claim_text: str) -> List[str]:
        """Rank phrases by specificity and overlap with claim content (generic)."""
        # prepare claim content tokens
        claim_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9'-]+", claim_text) if t.lower() not in STOP_WORDS]
        claim_set = set(claim_tokens)
        def score(p: str) -> float:
            toks = re.findall(r"[A-Za-z0-9'-]+", p)
            low = [t.lower() for t in toks]
            # overlap with claim content
            overlap = len(set(low) & claim_set)
            # proper-like tokens (Titlecase) indicate specificity
            proper_like = sum(1 for t in toks if t[:1].isupper() and t[1:].islower())
            # penalty for pronoun-led
            pron_pen = 1 if (low and low[0] in {'it','its','they','their','this','that'}) else 0
            # length moderation
            length_bonus = 1 if 3 <= len(toks) <= 10 else 0
            return overlap * 2 + proper_like * 1.5 + length_bonus - pron_pen * 2
        return sorted(phrases, key=score, reverse=True)

    def _simple_phrases(self, text: str) -> List[str]:
        """Fallback: higher-quality phrases using sentences and minimal context (generic)."""
        phrases: List[str] = []
        parts = re.split(r'[.!?]+', text)
        for part in parts:
            s = part.strip()
            if not s:
                continue
            # require a verb-like token to avoid noun-only fragments
            if not re.search(r"\b\w+(?:ed|ing)\b", s, flags=re.I):
                continue
            phrases.append(s)
            # add a short window around capitalized multi-token names
            for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', s):
                name = m.group(1)
                tail = s[m.end():].strip()
                tail_tokens = re.findall(r"[A-Za-z0-9'-]+", tail)
                tail_filtered = []
                for t in tail_tokens:
                    low = t.lower()
                    if low in STOP_WORDS or len(low) <= 2:
                        continue
                    tail_filtered.append(t)
                    if len(tail_filtered) >= 6:
                        break
                if tail_filtered:
                    phrase = f"{name} {' '.join(tail_filtered)}"
                    # ensure phrase has at least 3 tokens
                    if len(re.findall(r"[A-Za-z0-9'-]+", phrase)) >= 3:
                        phrases.append(phrase)
        # dedupe preserving order
        seen = set()
        out = []
        for p in phrases:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    async def _search_google(self, queries: List[str]) -> List[SearchResult]:
        """Search using Google Custom Search API."""
        results = []
        
        for query in queries:
            try:
                # Add delay to respect rate limits
                await asyncio.sleep(self.search_delay)
                
                search_params = {
                    'key': self.google_api_key,
                    'cx': '017576662512468239146:omuauf_lfve',  # Custom search engine ID
                    'q': query,
                    'num': 10
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://www.googleapis.com/customsearch/v1', params=search_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('items', []):
                                result = SearchResult(
                                    title=item.get('title', ''),
                                    url=item.get('link', ''),
                                    snippet=item.get('snippet', ''),
                                    source=item.get('displayLink', ''),
                                    published_date=None,
                                    relevance_score=0.0,
                                    domain=self._extract_domain(item.get('link', ''))
                                )
                                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in Google search: {e}")
        
        return results
    
    async def _search_news_api(self, queries: List[str]) -> List[SearchResult]:
        """Search using NewsAPI."""
        results = []
        
        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)
                
                # Search for articles
                articles = self.news_api.get_everything(
                    q=query,
                    language='en',
                    sort_by='relevancy',
                    page_size=10
                )
                
                for article in articles.get('articles', []):
                    result = SearchResult(
                        title=article.get('title', ''),
                        url=article.get('url', ''),
                        snippet=article.get('description', ''),
                        source=article.get('source', {}).get('name', ''),
                        published_date=article.get('publishedAt', ''),
                        relevance_score=0.0,
                        domain=self._extract_domain(article.get('url', ''))
                    )
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error in NewsAPI search: {e}")
        
        return results
    
    async def _search_serpapi(self, queries: List[str]) -> List[SearchResult]:
        """Search using SerpAPI."""
        results = []
        
        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)
                
                search = GoogleSearch({
                    "q": query,
                    "api_key": self.serpapi_key,
                    "num": 10
                })
                
                search_results = search.get_dict()
                
                for result in search_results.get('organic_results', []):
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=result.get('link', ''),
                        snippet=result.get('snippet', ''),
                        source=result.get('source', ''),
                        published_date=result.get('date', ''),
                        relevance_score=0.0,
                        domain=self._extract_domain(result.get('link', ''))
                    )
                    results.append(search_result)
                
            except Exception as e:
                logger.error(f"Error in SerpAPI search: {e}")
        
        return results
    
    async def _search_mediastack(self, queries: List[str]) -> List[SearchResult]:
        """Search using MediaStack API (free tier)."""
        results = []
        
        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)
                
                url = f"http://api.mediastack.com/v1/news"
                params = {
                    'access_key': self.mediastack_api_key,
                    'keywords': query,
                    'languages': 'en',
                    'limit': 10
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for article in data.get('data', []):
                                result = SearchResult(
                                    title=article.get('title', ''),
                                    url=article.get('url', ''),
                                    snippet=article.get('description', ''),
                                    source=article.get('source', ''),
                                    published_date=article.get('published_at', ''),
                                    relevance_score=0.0,
                                    domain=self._extract_domain(article.get('url', ''))
                                )
                                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in MediaStack search: {e}")
        
        return results
        
    async def _search_world_news_api(self, queries: List[str]) -> List[SearchResult]:
        """Search using WorldNewsApi.Com."""
        results = []
        base_url = "https://api.worldnewsapi.com/search-news"  # Real base URL

        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)

                params = {
                    'api-key': self.worldnewsapi_key,  # Note key name and dash
                    'text': query,
                    'language': 'en',
                    'page_size': 10
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.debug(f"WorldNewsAPI response for query '{query}': {data}")

                            # Check different possible response structures
                            articles = []
                            if 'news' in data:
                                articles = data['news']
                                logger.debug(f"Found {len(articles)} articles in 'news' key")
                            elif 'articles' in data:
                                articles = data['articles']
                                logger.debug(f"Found {len(articles)} articles in 'articles' key")
                            elif 'data' in data:
                                articles = data['data']
                                logger.debug(f"Found {len(articles)} articles in 'data' key")
                            else:
                                logger.warning(f"WorldNewsAPI: No articles found in response. Available keys: {list(data.keys())}")
                                continue

                            for article in articles:
                                # Try different field names for title
                                title = (article.get('title') or
                                        article.get('headline') or
                                        article.get('webTitle') or
                                        article.get('name') or '')

                                # Try different field names for URL
                                url = (article.get('url') or
                                      article.get('link') or
                                      article.get('webUrl') or '')

                                # Try different field names for description
                                snippet = (article.get('description') or
                                          article.get('summary') or
                                          article.get('snippet') or
                                          article.get('content') or '')

                                # Try different field names for source
                                source = (article.get('source_name') or
                                         article.get('source') or
                                         article.get('publisher') or
                                         article.get('provider', {}).get('name') or '')

                                # Try different field names for published date
                                published_date = (article.get('published_at') or
                                                 article.get('publishedAt') or
                                                 article.get('pubDate') or
                                                 article.get('date') or
                                                 article.get('publish_date') or '')

                                if title and url:  # Only add if we have at least title and URL
                                    result = SearchResult(
                                        title=title,
                                        url=url,
                                        snippet=snippet,
                                        source=source,
                                        published_date=published_date,
                                        relevance_score=0.0,
                                        domain=self._extract_domain(url)
                                    )
                                    results.append(result)
                                    logger.debug(f"Added article: {title[:50]}...")
                                else:
                                    logger.debug(f"Skipping article - missing title or URL: title='{title[:30]}...', url='{url[:30]}...'")

                        else:
                            logger.error(f"WorldNewsAPI query '{query}' failed with status {response.status}: {await response.text()}")

            except Exception as e:
                logger.error(f"Error in World News API search: {e}")

        logger.info(f"WorldNewsAPI: Total results collected: {len(results)}")
        return results

    async def _search_currents_api(self, queries: List[str]) -> List[SearchResult]:
        """Search using Currents API."""
        logger.info(f"Using Currents API key present: {bool(self.currents_api_key)}")
        results = []

        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)

                url = "https://api.currentsapi.services/v1/search"
                params = {
                    'apiKey': self.currents_api_key,
                    'keywords': query,
                    'language': 'en',
                    'limit': 10
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            for article in data.get('news', []):
                                result = SearchResult(
                                    title=article.get('title', ''),
                                    url=article.get('url', ''),
                                    snippet=article.get('description', ''),
                                    source=article.get('source', {}).get('name', ''),
                                    published_date=article.get('published', ''),
                                    relevance_score=0.0,
                                    domain=self._extract_domain(article.get('url', ''))
                                )
                                results.append(result)

            except Exception as e:
                logger.error(f"Error in Currents API search: {e}")

        return results

    async def _search_contextual_web_search(self, queries: List[str]) -> List[SearchResult]:
        """Search using Contextual Web Search API."""
        logger.info(f"Using Contextual Web Search API key present: {bool(self.contextual_web_search_api_key)}")
        results = []

        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)

                url = "https://contextualwebsearch-websearch-v1.p.rapidapi.com/api/search/NewsSearchAPI"
                headers = {
                    'X-RapidAPI-Key': self.contextual_web_search_api_key,
                    'X-RapidAPI-Host': 'contextualwebsearch-websearch-v1.p.rapidapi.com'
                }
                params = {
                    'q': query,
                    'pageNumber': 1,
                    'pageSize': 10,
                    'autoCorrect': 'true',
                    'safeSearch': 'true'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            for article in data.get('value', []):
                                result = SearchResult(
                                    title=article.get('title', ''),
                                    url=article.get('url', ''),
                                    snippet=article.get('description', ''),
                                    source=article.get('provider', {}).get('name', ''),
                                    published_date=article.get('datePublished', ''),
                                    relevance_score=0.0,
                                    domain=self._extract_domain(article.get('url', ''))
                                )
                                results.append(result)

            except Exception as e:
                logger.error(f"Error in Contextual Web Search: {e}")

        return results

    async def _search_the_guardian(self, queries: List[str]) -> List[SearchResult]:
        """Search using The Guardian News API."""
        results = []

        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)

                url = "https://content.guardianapis.com/search"
                params = {
                    'api-key': self.the_guardian_api_key,
                    'q': query,
                    'page-size': 10,
                    'show-fields': 'headline,trailText,byline,publication',
                    'order-by': 'relevance'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            for article in data.get('response', {}).get('results', []):
                                result = SearchResult(
                                    title=article.get('webTitle', ''),
                                    url=article.get('webUrl', ''),
                                    snippet=article.get('fields', {}).get('trailText', ''),
                                    source='The Guardian',
                                    published_date=article.get('webPublicationDate', ''),
                                    relevance_score=0.0,
                                    domain=self._extract_domain(article.get('webUrl', ''))
                                )
                                results.append(result)

            except Exception as e:
                logger.error(f"Error in The Guardian API search: {e}")

        return results


    async def _search_newsdata(self, queries: List[str]) -> List[SearchResult]:
        """Search using NewsData API (free tier)."""
        results = []
        
        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)
                
                url = f"https://newsdata.io/api/1/news"
                params = {
                    'apikey': self.newsdata_api_key,
                    'q': query,
                    'language': 'en',
                    'size': 10
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for article in data.get('results', []):
                                result = SearchResult(
                                    title=article.get('title', ''),
                                    url=article.get('link', ''),
                                    snippet=article.get('description', ''),
                                    source=article.get('source_id', ''),
                                    published_date=article.get('pubDate', ''),
                                    relevance_score=0.0,
                                    domain=self._extract_domain(article.get('link', ''))
                                )
                                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in NewsData search: {e}")
        
        return results
    
    async def _search_bing(self, queries: List[str]) -> List[SearchResult]:
        """Search using Bing Search API (free tier)."""
        results = []
        
        for query in queries:
            try:
                await asyncio.sleep(self.search_delay)
                
                url = "https://api.bing.microsoft.com/v7.0/search"
                headers = {
                    'Ocp-Apim-Subscription-Key': self.bing_api_key
                }
                params = {
                    'q': query,
                    'count': 10,
                    'mkt': 'en-US'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('webPages', {}).get('value', []):
                                result = SearchResult(
                                    title=item.get('name', ''),
                                    url=item.get('url', ''),
                                    snippet=item.get('snippet', ''),
                                    source=item.get('displayUrl', ''),
                                    published_date=None,
                                    relevance_score=0.0,
                                    domain=self._extract_domain(item.get('url', ''))
                                )
                                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in Bing search: {e}")
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ""
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL."""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

    def _apply_robust_pre_filters(self, results: List[SearchResult], claim_text: str) -> List[SearchResult]:
        """Apply two-stage pre-filtering with fallback to ensure scoring always runs."""
        initial_count = len(results)
        logger.info(f"Starting robust pre-filtering: {initial_count} results")

        # Stage 1: Basic quality filters
        stage1_results = []
        for result in results:
            # Skip empty domains
            if not result.domain or not result.domain.strip():
                continue

            # Skip blacklisted paths
            url_lower = (result.url or "").lower()
            if any(path in url_lower for path in self.blacklisted_paths):
                continue

            # Skip non-article content
            if any(exclude in url_lower for exclude in ['/video', '/image', '/gallery', '/podcast']):
                continue

            stage1_results.append(result)

        logger.info(f"Stage 1 (quality): {initial_count} -> {len(stage1_results)} results")

        # Stage 2: Entity and topic relevance filters
        stage2_results = []
        for result in stage1_results:
            # Compute lightweight signals
            entity_coverage = self._compute_entity_coverage(result)
            keyword_overlap = self._compute_keyword_overlap(result)

            # Primary pass: strict requirements
            if entity_coverage >= 0.5 and keyword_overlap >= 0.2:
                stage2_results.append(result)
                continue

            # Fallback pass: softer requirements
            if entity_coverage >= 0.5 or keyword_overlap >= 0.3:
                stage2_results.append(result)
                continue

            # Final fallback: lowered threshold to retain more
            if entity_coverage > 0.1 or keyword_overlap > 0.1:
                stage2_results.append(result)

        logger.info(f"Stage 2 (relevance): {len(stage1_results)} -> {len(stage2_results)} results")

        # Stage 3: Source credibility (only if we have candidates)
        if stage2_results:
            final_results = []
            for result in stage2_results:
                try:
                    credibility = self.source_checker.analyze_source(result.url).credibility_score
                    if credibility >= 0.5:  # Require reliable sources
                        final_results.append(result)
                except Exception as e:
                    logger.warning(f"Credibility check failed for {result.domain}: {e}")
                    # Include on error to avoid over-filtering
                    final_results.append(result)

            logger.info(f"Stage 3 (credibility): {len(stage2_results)} -> {len(final_results)} results")
            return final_results

        # Emergency fallback: if no results pass filters, return top 5 by domain reliability
        logger.warning("All results failed pre-filters, using emergency fallback")
        fallback_results = []
        for result in results[:10]:  # Check top 10
            try:
                credibility = self.source_checker.analyze_source(result.url).credibility_score
                if credibility >= 0.6:  # Higher threshold for fallback
                    fallback_results.append(result)
                    if len(fallback_results) >= 5:
                        break
            except:
                continue

        logger.info(f"Emergency fallback: {len(fallback_results)} results")
        return fallback_results
    
    def _rank_results_with_calibrated_scoring(self, results: List[SearchResult], claim_text: str) -> List[SearchResult]:
        """Rank results with calibrated relevance scoring that always produces non-zero scores."""
        for result in results:
            result.relevance_score = self._calculate_calibrated_relevance_score(result, claim_text)

        # Sort by relevance score descending
        ranked = sorted(results, key=lambda x: x.relevance_score, reverse=True)

        # Ensure all_retrieved_articles gets the scored list
        logger.info(f"Ranked {len(ranked)} results with calibrated scores")
        return ranked

    def _calculate_calibrated_relevance_score(self, result: SearchResult, claim_text: str) -> float:
        """Calculate calibrated relevance score with guaranteed non-zero values."""
        # Encode claim text
        claim_embedding = self.embedding_model.encode(claim_text)

        # Encode title and snippet if available
        title_sim = 0.0
        if result.title:
            title_embedding = self.embedding_model.encode(result.title)
            title_sim = util.cos_sim(claim_embedding, title_embedding).item()

        snippet_sim = 0.0
        if result.snippet:
            snippet_embedding = self.embedding_model.encode(result.snippet)
            snippet_sim = util.cos_sim(claim_embedding, snippet_embedding).item()

        # Fallback similarity using domain/source if title/snippet missing
        fallback_sim = 0.0
        if not result.title and not result.snippet:
            domain_text = f"{result.domain} {result.source}"
            if domain_text.strip():
                domain_embedding = self.embedding_model.encode(domain_text)
                fallback_sim = util.cos_sim(claim_embedding, domain_embedding).item() * 0.3

        # Use reliable domains from source checker
        credible_domains = getattr(self.source_checker, 'reliable_sources', {
            'bbc.com', 'reuters.com', 'ap.org', 'npr.org', 'cnn.com', 'nytimes.com'
        })

        domain_lower = (result.domain or "").lower()

        score = 0.1  # Base score to ensure non-zero
        score += title_sim * 0.5
        score += snippet_sim * 0.3
        score += fallback_sim
        if domain_lower in credible_domains:
            score += 0.2

        if result.published_date:
            try:
                pub_date = datetime.fromisoformat(result.published_date.replace('Z', '+00:00'))
                days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
                if days_old < 7:
                    score += 0.15  # Increased recency bonus
            except Exception as e:
                logger.warning(f"Failed to parse published_date {result.published_date}: {e}")

        # Normalize to [0.1, 1.0]
        score = max(0.1, min(score, 1.0))

        return score

    def _compute_entity_coverage(self, result: SearchResult) -> float:
        """Compute entity coverage score based on overlap with extracted entities."""
        if not self.entity_anchors and not self.topic_tokens and not self.numeric_patterns:
            return 0.0

        # Combine title and snippet for analysis
        text = f"{result.title} {result.snippet}".lower()
        words = set(re.findall(r'\b\w+\b', text))

        # Calculate overlaps
        entity_overlap = len(words & self.entity_anchors)
        topic_overlap = len(words & self.topic_tokens)
        numeric_overlap = len(words & self.numeric_patterns)

        # Total possible matches
        total_entities = len(self.entity_anchors) + len(self.topic_tokens) + len(self.numeric_patterns)
        if total_entities == 0:
            return 0.0

        # Weighted coverage (entities most important)
        coverage = (entity_overlap * 2 + topic_overlap + numeric_overlap) / (total_entities * 2)
        return min(coverage, 1.0)

    def _compute_keyword_overlap(self, result: SearchResult) -> float:
        """Compute keyword overlap score using Jaccard similarity with claim keywords."""
        if not hasattr(self, 'claim_text') or not self.claim_text:
            return 0.0

        # Extract claim words, excluding stopwords
        claim_lower = self.claim_text.lower()
        claim_words = set(re.findall(r'\b\w+\b', claim_lower)) - STOP_WORDS

        if not claim_words:
            return 0.0

        # Combine title and snippet
        text = f"{result.title} {result.snippet}".lower()
        result_words = set(re.findall(r'\b\w+\b', text)) - STOP_WORDS

        # Jaccard similarity
        intersection = len(claim_words & result_words)
        union = len(claim_words | result_words)
        if union == 0:
            return 0.0

        return intersection / union
    
    async def _fetch_content_for_top_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Fetch full content for top search results."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for result in results:
                task = self._fetch_article_content(session, result)
                tasks.append(task)
            
            updated_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = []
            for result in updated_results:
                if isinstance(result, SearchResult):
                    valid_results.append(result)
            
            return valid_results
    
    async def _fetch_article_content(self, session: aiohttp.ClientSession, result: SearchResult) -> SearchResult:
        """Fetch full content of an article."""
        try:
            async with session.get(result.url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract main content
                    content_selectors = [
                        'article', '.article-content', '.post-content',
                        '.entry-content', '.content', 'main'
                    ]
                    
                    content = ""
                    for selector in content_selectors:
                        elements = soup.select(selector)
                        if elements:
                            content = ' '.join([elem.get_text() for elem in elements])
                            break
                    
                    if not content:
                        # Fallback: get all paragraph text
                        paragraphs = soup.find_all('p')
                        content = ' '.join([p.get_text() for p in paragraphs])
                    
                    result.content = content[:2000]  # Limit content length
                    
        except Exception as e:
            logger.error(f"Error fetching content from {result.url}: {e}")
        
        return result

# Example usage
if __name__ == "__main__":
    async def main():
        search_engine = WebSearchEngine()
        
        claim = "The moon landing was faked by NASA in 1969."
        evidence = await search_engine.search_for_evidence(claim)
        
        print(f"Found {evidence.total_results} results for claim: {evidence.claim_text}")
        for i, result in enumerate(evidence.search_results[:3]):
            print(f"\n{i+1}. {result.title}")
            print(f"   Source: {result.source}")
            print(f"   Relevance: {result.relevance_score:.2f}")
            print(f"   URL: {result.url}")
    
    asyncio.run(main())
