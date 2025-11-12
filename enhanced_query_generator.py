"""
Enhanced Query Generation Module

This module implements advanced query generation strategies including:
- Semantic similarity grouping
- Query performance tracking
- Temporal context awareness
- Claim type classification
- Multilingual support
"""

import os
import re
from typing import List, Dict, Set, Tuple
from datetime import datetime
import json
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from langdetect import detect
from googletrans import Translator
import spacy
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryPerformanceTracker:
    """Tracks and stores query performance metrics."""
    
    def __init__(self, storage_path: str = "data/query_performance.json"):
        self.storage_path = storage_path
        self.performance_data = self._load_performance_data()
        
    def _load_performance_data(self) -> Dict:
        """Load existing performance data from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
        return {"queries": {}, "patterns": {}}
    
    def update_query_performance(self, query: str, success_score: float):
        """Update performance metrics for a query."""
        if query not in self.performance_data["queries"]:
            self.performance_data["queries"][query] = {
                "total_uses": 0,
                "success_sum": 0,
                "avg_success": 0
            }
        
        data = self.performance_data["queries"][query]
        data["total_uses"] += 1
        data["success_sum"] += success_score
        data["avg_success"] = data["success_sum"] / data["total_uses"]
        
        # Update pattern statistics
        pattern = self._extract_query_pattern(query)
        if pattern:
            if pattern not in self.performance_data["patterns"]:
                self.performance_data["patterns"][pattern] = {
                    "total_uses": 0,
                    "success_sum": 0,
                    "avg_success": 0
                }
            
            pattern_data = self.performance_data["patterns"][pattern]
            pattern_data["total_uses"] += 1
            pattern_data["success_sum"] += success_score
            pattern_data["avg_success"] = pattern_data["success_sum"] / pattern_data["total_uses"]
        
        self._save_performance_data()
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract the general pattern from a query."""
        # Replace specific terms with placeholders while keeping structure
        pattern = query.lower()
        pattern = re.sub(r'\b\d+\b', 'NUM', pattern)
        pattern = re.sub(r'\b[a-z]+ed\b', 'VERB', pattern)
        pattern = re.sub(r'\b[A-Z][a-z]+\b', 'ENTITY', pattern)
        return pattern
    
    def get_best_patterns(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get the most successful query patterns."""
        patterns = self.performance_data["patterns"]
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: (x[1]["avg_success"], x[1]["total_uses"]),
            reverse=True
        )
        return sorted_patterns[:top_n]
    
    def _save_performance_data(self):
        """Save performance data to storage."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
            
    def get_query_performance(self, query: str) -> float:
        """Get the average success score for a query."""
        if query in self.performance_data["queries"]:
            return self.performance_data["queries"][query]["avg_success"]
        return 0.0

class ClaimClassifier:
    """Classifies claims into different types for targeted query generation."""
    
    def __init__(self):
        """Initialize claim classifier with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")
            self.nlp = None
        
        self.claim_types = {
            "SCIENTIFIC": r"\b(study|research|scientists?|evidence|data|experiment)\b",
            "POLITICAL": r"\b(government|election|president|congress|vote|politician)\b",
            "HEALTH": r"\b(health|medical|disease|cure|vaccine|treatment)\b",
            "CONSPIRACY": r"\b(conspiracy|secret|cover[- ]up|hoax|illuminati)\b",
            "HISTORICAL": r"\b(\d{4}|\d{2}s|history|ancient|century|era)\b",
            "CELEBRITY": r"\b(star|actor|actress|singer|famous|celebrity)\b"
        }
    
    def classify_claim(self, claim_text: str) -> List[str]:
        """Classify a claim into one or more types."""
        claim_types = []
        
        # Check each type's patterns
        for claim_type, pattern in self.claim_types.items():
            if re.search(pattern, claim_text, re.IGNORECASE):
                claim_types.append(claim_type)
        
        # Use spaCy for additional analysis if available
        if self.nlp:
            doc = self.nlp(claim_text)
            
            # Check for named entities
            entity_types = set(ent.label_ for ent in doc.ents)
            
            if "PERSON" in entity_types:
                claim_types.append("PERSON_RELATED")
            if "ORG" in entity_types:
                claim_types.append("ORGANIZATION_RELATED")
            if "GPE" in entity_types or "LOC" in entity_types:
                claim_types.append("LOCATION_RELATED")
            if "DATE" in entity_types or "TIME" in entity_types:
                claim_types.append("TEMPORAL")
        
        return list(set(claim_types))

class EnhancedQueryGenerator:
    """Enhanced query generation with advanced features."""
    
    def __init__(self):
        """Initialize the enhanced query generator."""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.performance_tracker = QueryPerformanceTracker()
        self.claim_classifier = ClaimClassifier()
        self.translator = Translator()
        
        # Load temporal keywords
        self.temporal_keywords = {
            "recent": ["latest", "recent", "new", "current", "ongoing"],
            "historical": ["historical", "history", "archive", "past", "original"],
            "specific_date": []  # Will be populated based on claim dates
        }
    
    async def generate_queries(self, claim_text: str, languages: List[str] = None) -> List[Dict[str, str]]:
        """Generate enhanced queries for the claim."""
        if not claim_text:
            return []
        
        # Default to English if no languages specified
        if not languages:
            languages = ['en']
        
        # Detect claim language if not specified
        try:
            source_lang = detect(claim_text)
        except:
            source_lang = 'en'
        
        # 1. Generate base queries
        base_queries = self._generate_base_queries(claim_text)
        
        # 2. Add temporal context
        temporal_queries = self._add_temporal_context(claim_text)
        
        # 3. Classify claim and add targeted queries
        claim_types = self.claim_classifier.classify_claim(claim_text)
        targeted_queries = self._generate_targeted_queries(claim_text, claim_types)
        
        # 4. Combine all queries and remove duplicates
        all_queries = list({q.strip() for q in (base_queries + temporal_queries + targeted_queries) if q.strip()})
        
        # Add original language queries first
        multilingual_queries = [{"query": q, "language": source_lang} for q in all_queries]
        used_queries = {q.strip().lower() for q in all_queries}
        
        # 5. Group similar queries using semantic similarity
        grouped_queries = self._group_similar_queries(all_queries)
        
        # 6. Generate multilingual variations
        for query_group in grouped_queries:
            # Take the best performing query from each group
            best_query = self._select_best_query(query_group)
            if best_query:
                # Only translate if we haven't seen this query before
                query_key = best_query.lower().strip()
                if query_key not in used_queries:
                    used_queries.add(query_key)
                    # Only translate to other languages
                    other_langs = [l for l in languages if l != source_lang]
                    if other_langs:
                        translated_queries = await self._translate_query(best_query, source_lang, other_langs)
                        # Filter out any corrupted translations
                        valid_queries = [
                            q for q in translated_queries
                            if self._is_valid_translation(q["query"]) and
                            q["query"].strip().lower() not in used_queries
                        ]
                        for q in valid_queries:
                            used_queries.add(q["query"].strip().lower())
                        multilingual_queries.extend(valid_queries)
        
        return multilingual_queries
    
    def _generate_base_queries(self, claim_text: str) -> List[str]:
        """Generate base queries with performance-tracked patterns."""
        queries = set()  # Use set to avoid duplicates
        
        # Clean trailing punctuation
        claim_clean = re.sub(r'[\.\!\?]+$', '', claim_text.strip())
        
        # Add exact claim and quoted version
        queries.add(claim_text.strip())
        queries.add(f'"{claim_clean}"')
        
        # Add core claim version
        core_claim = self._extract_core_claim(claim_text)
        core_clean = re.sub(r'[\.\!\?]+$', '', core_claim.strip())
        if core_claim != claim_text:
            queries.add(core_claim.strip())
            queries.add(f'"{core_clean}"')
            
        # Add fact-checking specific queries
        queries.add(f"fact check {core_claim}")
        queries.add(f"verify {core_claim}")
        queries.add(f"debunk {core_claim}")
        
        # Add source verification queries
        entities = self._extract_entities(claim_text)
        if entities and entities != claim_text:
            queries.add(f"official statement {entities}")
            queries.add(f"verified source {entities}")
        
        # Add queries based on best performing patterns
        best_patterns = self.performance_tracker.get_best_patterns(3)  # Limit to top 3 patterns
        for pattern, stats in best_patterns:
            if stats["avg_success"] > 0.5:  # Only use patterns with good success rate
                try:
                    query = self._apply_pattern(claim_text, pattern)
                    if query and self._is_valid_translation(query):  # Reuse translation validator
                        queries.add(query.strip())
                except Exception as e:
                    logger.error(f"Error applying pattern {pattern}: {e}")
        
        return list(queries)
    
    def _add_temporal_context(self, claim_text: str) -> List[str]:
        """Add temporal context to queries."""
        temporal_queries = []
        
        # Extract dates and temporal references
        dates = re.findall(r'\b\d{4}\b', claim_text)
        today_words = ['today', 'yesterday', 'this morning', 'this evening', 'last night']
        temporal_refs = []
        
        # Check for temporal references
        claim_lower = claim_text.lower()
        for keyword in self.temporal_keywords["recent"] + today_words:
            if keyword in claim_lower:
                temporal_refs.append(keyword)
        
        base_query = self._extract_core_claim(claim_text)
        
        if dates:
            # Add year-specific variations
            for date in dates:
                temporal_queries.append(f"{base_query} from {date}")
                temporal_queries.append(f"{base_query} in {date}")
                self.temporal_keywords["specific_date"].extend([
                    f"before {date}",
                    f"after {date}",
                    f"during {date}",
                    f"in {date}"
                ])
        
        if temporal_refs:
            # Add recency-based variations
            for ref in temporal_refs:
                temporal_queries.append(f"{base_query} {ref}")
                temporal_queries.append(f"recent {base_query}")
                temporal_queries.append(f"latest {base_query}")
        
        if not (dates or temporal_refs):
            # If no temporal references found, add a general time-based query
            temporal_queries.append(f"when did {base_query}")
            temporal_queries.append(f"recent {base_query}")
        
        return list(set(temporal_queries))  # Remove duplicates
    
    def _generate_targeted_queries(self, claim_text: str, claim_types: List[str]) -> List[str]:
        """Generate queries targeted to specific claim types."""
        targeted_queries = []
        base_query = self._extract_core_claim(claim_text)
        base_clean = re.sub(r'[\.\!\?]+$', '', base_query.strip())
        
        type_specific_keywords = {
            "SCIENTIFIC": [
                "scientific evidence for",
                "research about",
                "studies on",
                "scientific consensus"
            ],
            "POLITICAL": [
                "fact check",
                "official sources on",
                "government statement about",
                "verified information"
            ],
            "HEALTH": [
                "medical evidence for",
                "health authority statement",
                "clinical research on",
                "medical facts about"
            ],
            "CONSPIRACY": [
                "evidence about",
                "investigation of",
                "proof regarding",
                "fact check"
            ],
            "HISTORICAL": [
                "historical evidence of",
                "historical records about",
                "documentation of",
                "archive evidence"
            ],
            "CELEBRITY": [
                "official statement about",
                "verified news about",
                "reliable source on",
                "fact check"
            ],
            "PERSON_RELATED": [
                "verified information about",
                "official statement from",
                "reliable source on",
                "fact check"
            ],
            "ORGANIZATION_RELATED": [
                "official statement from",
                "company announcement about",
                "organization verification",
                "fact check"
            ],
            "LOCATION_RELATED": [
                "local news about",
                "verified reports from",
                "official sources on",
                "fact check"
            ],
            "TEMPORAL": [
                "timeline of",
                "chronology of",
                "date verification for",
                "fact check"
            ]
        }
        
        used_queries = set()
        for claim_type in claim_types:
            if claim_type in type_specific_keywords:
                keywords = type_specific_keywords[claim_type]
                for keyword in keywords:
                    query = f"{keyword} {base_query}".strip()
                    if query not in used_queries:
                        used_queries.add(query)
                        targeted_queries.append(query)
                        
                        # Add a quoted version for exact matches
                        quoted_query = f'{keyword} "{base_clean}"'
                        if quoted_query not in used_queries:
                            used_queries.add(quoted_query)
                            targeted_queries.append(quoted_query)
        
        return list(set(targeted_queries))  # Remove any remaining duplicates
    
    def _group_similar_queries(self, queries: List[str]) -> List[List[str]]:
        """Group semantically similar queries."""
        if not queries:
            return []
        
        # Generate embeddings for all queries
        embeddings = self.embedding_model.encode(queries)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.3,  # Adjust threshold as needed
            metric='cosine',
            linkage='average'
        )
        
        clustering.fit(embeddings)
        
        # Group queries by cluster
        groups = defaultdict(list)
        for i, label in enumerate(clustering.labels_):
            groups[label].append(queries[i])
        
        return list(groups.values())
    
    def _select_best_query(self, query_group: List[str]) -> str:
        """Select the best performing query from a group."""
        best_query = query_group[0]
        best_score = 0
        
        for query in query_group:
            if query in self.performance_tracker.performance_data["queries"]:
                score = self.performance_tracker.performance_data["queries"][query]["avg_success"]
                if score > best_score:
                    best_score = score
                    best_query = query
        
        return best_query
    
    async def _translate_query(self, query: str, source_lang: str, target_langs: List[str]) -> List[Dict[str, str]]:
        """Translate a query into multiple languages."""
        translations = []
        
        for target_lang in target_langs:
            if target_lang == source_lang:
                translations.append({
                    "query": query,
                    "language": source_lang
                })
                continue
                
            try:
                translated = await self.translator.translate(query, src=source_lang, dest=target_lang)
                if hasattr(translated, 'text'):
                    translations.append({
                        "query": translated.text,
                        "language": target_lang
                    })
                else:
                    logger.warning(f"Translation result for {target_lang} has no 'text' attribute")
            except Exception as e:
                logger.error(f"Translation error for {target_lang}: {e}")
        
        return translations
    
    def _apply_pattern(self, claim_text: str, pattern: str) -> str:
        """Apply a query pattern to the claim text."""
        # Replace placeholders with actual content
        query = pattern
        query = query.replace("CLAIM", claim_text)
        query = query.replace("ENTITY", self._extract_entities(claim_text))
        query = query.replace("VERB", self._extract_verbs(claim_text))
        return query
    
    def _extract_entities(self, text: str) -> str:
        """Extract named entities from text."""
        if self.claim_classifier.nlp:
            doc = self.claim_classifier.nlp(text)
            entities = [ent.text for ent in doc.ents]
            return " ".join(entities) if entities else text
        return text
    
    def _extract_verbs(self, text: str) -> str:
        """Extract main verbs from text."""
        if self.claim_classifier.nlp:
            doc = self.claim_classifier.nlp(text)
            verbs = [token.text for token in doc if token.pos_ == "VERB"]
            return " ".join(verbs) if verbs else text
        return text
        
    def _extract_core_claim(self, text: str) -> str:
        """Extract the core claim from the text by focusing on key entities and actions."""
        if not self.claim_classifier.nlp:
            return text

        doc = self.claim_classifier.nlp(text)

        # For short claims (< 50 chars), return as-is to preserve meaning
        if len(text) < 50:
            return text

        # Extract key components while preserving sentence structure
        entities = [ent.text for ent in doc.ents]

        # Get main verbs that are not auxiliary
        main_verbs = []
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ not in ['be', 'have', 'do', 'is', 'are', 'was', 'were']:
                main_verbs.append(token.text)

        # If we have both entities and verbs, try to construct a meaningful core claim
        if entities and main_verbs:
            # Find the main subject (person/entity that is the subject)
            subjects = []
            for token in doc:
                if token.dep_ in ['nsubj', 'nsubjpass'] and token.pos_ in ['PROPN', 'NOUN']:
                    # For proper nouns, include compound parts (full names)
                    subject_parts = []
                    for t in token.subtree:
                        if t.dep_ in ['compound', 'nsubj', 'nsubjpass'] and t.pos_ in ['PROPN', 'NOUN']:
                            subject_parts.append(t.text)
                    if subject_parts:
                        subjects.append(" ".join(subject_parts))

            # If we have a clear subject-verb structure, preserve it
            if subjects:
                subject = subjects[0]  # Take first subject
                verb = main_verbs[0] if main_verbs else ""

                # Add key descriptive elements (objects, attributes)
                descriptors = []
                for token in doc:
                    if token.dep_ in ['dobj', 'attr', 'acomp'] and token.pos_ in ['NOUN', 'ADJ', 'PROPN']:
                        # Include subtree for compound descriptions
                        desc_parts = []
                        for t in token.subtree:
                            if t.pos_ in ['NOUN', 'ADJ', 'PROPN', 'NUM', 'DET']:
                                desc_parts.append(t.text)
                        if desc_parts:
                            descriptors.append(" ".join(desc_parts))

                # Add prepositional phrases that are important
                prep_phrases = []
                for token in doc:
                    if token.dep_ == 'prep' and token.text.lower() in ['with', 'of', 'in', 'at']:
                        prep_obj = []
                        for child in token.children:
                            if child.dep_ == 'pobj':
                                for t in child.subtree:
                                    if t.pos_ in ['NOUN', 'ADJ', 'PROPN', 'NUM']:
                                        prep_obj.append(t.text)
                        if prep_obj:
                            prep_phrases.append(f"{token.text} {' '.join(prep_obj)}")

                core_parts = [subject]
                if verb:
                    core_parts.append(verb)
                core_parts.extend(descriptors[:2])  # Limit descriptors
                core_parts.extend(prep_phrases[:1])  # Limit prep phrases

                core_claim = " ".join(core_parts)
                # Only return if it's reasonably shorter than original and meaningful
                if (len(core_claim) >= 15 and len(core_claim) < len(text) * 0.9 and
                    len(core_claim.split()) >= 3):
                    return core_claim

        # Fallback: return original text if we can't meaningfully shorten it
        return text
        
    def _is_valid_translation(self, text: str) -> bool:
        """Check if a translation appears valid and not corrupted."""
        if not text or len(text.strip()) == 0:
            return False
            
        # Check for common signs of corrupted translations
        red_flags = [
            text.count('{') > 0,  # JSON artifacts
            text.count('}') > 0,
            text.count('[') > 0,
            text.count(']') > 0,
            len(text.split()) < 2,  # Too short
            text.count('.') > 3,  # Too many periods
            text.isupper(),  # All caps
            text.islower(),  # All lowercase
            len(set(text)) < 5  # Too few unique characters
        ]
        
        return not any(red_flags)

# Example usage
async def main():
    generator = EnhancedQueryGenerator()
    
    # Test claim
    claim = "NASA faked the moon landing in 1969"
    
    # Generate queries in multiple languages
    queries = await generator.generate_queries(claim, languages=['en', 'es', 'fr'])
    
    # Print results
    print(f"\nGenerated queries for claim: {claim}")
    print("-" * 50)
    for query_data in queries:
        print(f"Language: {query_data['language']}")
        print(f"Query: {query_data['query']}")
        print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())