"""
Cross-Check Verification Module

This module uses NLP models to perform textual entailment and cross-check
claims against retrieved evidence.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationResult:
    """Represents the result of claim verification."""
    claim: str
    evidence_text: str
    entailment_score: float  # 0.0 to 1.0
    contradiction_score: float  # 0.0 to 1.0
    neutral_score: float  # 0.0 to 1.0
    overall_verdict: str  # 'SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO'
    confidence: float
    reasoning: str

@dataclass
class CrossCheckAnalysis:
    """Represents comprehensive cross-check analysis."""
    claim: str
    verification_results: List[VerificationResult]
    overall_support_score: float
    overall_refute_score: float
    overall_neutral_score: float
    final_verdict: str
    confidence: float
    evidence_count: int
    reliable_evidence_count: int

class CrossCheckVerifier:
    """Handles cross-check verification using NLP models."""
    
    def __init__(self):
        """Initialize the cross-check verifier with required models."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize textual entailment model
        self.entailment_model_name = "facebook/bart-large-mnli"
        try:
            self.entailment_tokenizer = AutoTokenizer.from_pretrained(self.entailment_model_name)
            self.entailment_model = AutoModelForSequenceClassification.from_pretrained(self.entailment_model_name)
            self.entailment_model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not load entailment model: {e}")
            self.entailment_model = None
            self.entailment_tokenizer = None
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            logger.warning(f"Could not load spaCy model: {e}")
            self.nlp = None
        
        # Initialize fact-checking pipeline
        try:
            self.fact_checker = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load fact-checking pipeline: {e}")
            self.fact_checker = None
        
        logger.info("Cross-check verifier initialized")
    
    def verify_claim_against_evidence(self, claim: str, evidence_text: str) -> VerificationResult:
        """Verify a claim against a single piece of evidence."""
        try:
            # Clean and preprocess texts
            claim_clean = self._clean_text(claim)
            evidence_clean = self._clean_text(evidence_text)
            
            # Calculate semantic similarity
            similarity_score = self._calculate_semantic_similarity(claim_clean, evidence_clean)
            
            # Perform textual entailment analysis
            entailment_scores = self._analyze_textual_entailment(claim_clean, evidence_clean)
            
            # Analyze contradiction
            contradiction_score = self._analyze_contradiction(claim_clean, evidence_clean)
            
            # Calculate overall scores
            support_score = entailment_scores.get('entailment', 0.0)
            refute_score = contradiction_score
            neutral_score = entailment_scores.get('neutral', 0.0)
            
            # Determine verdict
            verdict, confidence = self._determine_verdict(support_score, refute_score, neutral_score)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(claim_clean, evidence_clean, support_score, refute_score, similarity_score)
            
            return VerificationResult(
                claim=claim,
                evidence_text=evidence_text,
                entailment_score=support_score,
                contradiction_score=refute_score,
                neutral_score=neutral_score,
                overall_verdict=verdict,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error verifying claim against evidence: {e}")
            return VerificationResult(
                claim=claim,
                evidence_text=evidence_text,
                entailment_score=0.0,
                contradiction_score=0.0,
                neutral_score=1.0,
                overall_verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                reasoning=f"Error in verification: {str(e)}"
            )
    
    def cross_check_claim(self, claim: str, evidence_list: List[Dict]) -> CrossCheckAnalysis:
        """Perform comprehensive cross-check analysis of a claim against multiple evidence sources."""
        try:
            verification_results = []
            support_scores = []
            refute_scores = []
            neutral_scores = []
            reliable_evidence_count = 0
            
            for evidence in evidence_list:
                evidence_text = evidence.get('content', '') or evidence.get('snippet', '')
                source_reliability = evidence.get('source_reliability_score', 0.5)

                if evidence_text.strip():
                    # Verify claim against this evidence
                    result = self.verify_claim_against_evidence(claim, evidence_text)
                    verification_results.append(result)

                    # Weight scores by source reliability
                    weighted_support = result.entailment_score * source_reliability
                    weighted_refute = result.contradiction_score * source_reliability
                    weighted_neutral = result.neutral_score * source_reliability

                    support_scores.append(weighted_support)
                    refute_scores.append(weighted_refute)
                    neutral_scores.append(weighted_neutral)

                    if source_reliability >= 0.5:
                        reliable_evidence_count += 1
            
            # Calculate overall scores
            overall_support = np.mean(support_scores) if support_scores else 0.0
            overall_refute = np.mean(refute_scores) if refute_scores else 0.0
            overall_neutral = np.mean(neutral_scores) if neutral_scores else 0.0
            
            # Determine final verdict
            final_verdict, confidence = self._determine_final_verdict(
                overall_support, overall_refute, overall_neutral, len(verification_results)
            )
            
            return CrossCheckAnalysis(
                claim=claim,
                verification_results=verification_results,
                overall_support_score=overall_support,
                overall_refute_score=overall_refute,
                overall_neutral_score=overall_neutral,
                final_verdict=final_verdict,
                confidence=confidence,
                evidence_count=len(verification_results),
                reliable_evidence_count=reliable_evidence_count
            )
            
        except Exception as e:
            logger.error(f"Error in cross-check analysis: {e}")
            return CrossCheckAnalysis(
                claim=claim,
                verification_results=[],
                overall_support_score=0.0,
                overall_refute_score=0.0,
                overall_neutral_score=1.0,
                final_verdict="NOT_ENOUGH_INFO",
                confidence=0.0,
                evidence_count=0,
                reliable_evidence_count=0
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            if self.sentence_model is None:
                return 0.5  # Return neutral similarity if model not available
            
            # Encode texts
            embeddings = self.sentence_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.5
    
    def _analyze_textual_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Analyze textual entailment between premise and hypothesis."""
        try:
            if self.entailment_model is None or self.entailment_tokenizer is None:
                # Fallback to simple heuristic
                return self._heuristic_entailment_analysis(premise, hypothesis)
            
            # Prepare input
            inputs = self.entailment_tokenizer(
                premise, hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.entailment_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract scores (assuming 3 classes: entailment, neutral, contradiction)
            scores = probabilities[0].cpu().numpy()
            
            return {
                'entailment': float(scores[0]),
                'neutral': float(scores[1]),
                'contradiction': float(scores[2])
            }
            
        except Exception as e:
            logger.error(f"Error in textual entailment analysis: {e}")
            return self._heuristic_entailment_analysis(premise, hypothesis)
    
    def _heuristic_entailment_analysis(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Fallback heuristic analysis for textual entailment."""
        # Simple keyword-based analysis
        premise_words = set(premise.lower().split())
        hypothesis_words = set(hypothesis.lower().split())
        
        # Calculate overlap
        overlap = len(premise_words.intersection(hypothesis_words))
        total_hypothesis_words = len(hypothesis_words)
        
        if total_hypothesis_words == 0:
            return {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
        
        overlap_ratio = overlap / total_hypothesis_words
        
        # Simple scoring based on overlap
        if overlap_ratio > 0.7:
            return {'entailment': 0.8, 'neutral': 0.1, 'contradiction': 0.1}
        elif overlap_ratio > 0.3:
            return {'entailment': 0.3, 'neutral': 0.6, 'contradiction': 0.1}
        else:
            return {'entailment': 0.1, 'neutral': 0.7, 'contradiction': 0.2}
    
    def _analyze_contradiction(self, text1: str, text2: str) -> float:
        """Analyze contradiction between two texts."""
        try:
            if self.nlp is None:
                return self._heuristic_contradiction_analysis(text1, text2)
            
            # Extract entities and key concepts
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Get entities
            entities1 = set(ent.text.lower() for ent in doc1.ents)
            entities2 = set(ent.text.lower() for ent in doc2.ents)
            
            # Check for contradictory entities
            contradiction_score = 0.0
            
            # Simple contradiction patterns
            contradiction_patterns = [
                (r'\b(?:not|no|never|none|nothing)\b', r'\b(?:is|are|was|were|has|have|had)\b'),
                (r'\b(?:true|real|actual|confirmed)\b', r'\b(?:false|fake|untrue|denied)\b'),
                (r'\b(?:increased|rose|grew)\b', r'\b(?:decreased|fell|dropped)\b'),
                (r'\b(?:support|agree|endorse)\b', r'\b(?:oppose|disagree|reject)\b')
            ]
            
            for pattern1, pattern2 in contradiction_patterns:
                if re.search(pattern1, text1.lower()) and re.search(pattern2, text2.lower()):
                    contradiction_score += 0.3
                elif re.search(pattern2, text1.lower()) and re.search(pattern1, text2.lower()):
                    contradiction_score += 0.3
            
            return min(contradiction_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing contradiction: {e}")
            return self._heuristic_contradiction_analysis(text1, text2)
    
    def _heuristic_contradiction_analysis(self, text1: str, text2: str) -> float:
        """Fallback heuristic contradiction analysis."""
        # Simple keyword-based contradiction detection
        contradiction_keywords = {
            'not', 'no', 'never', 'none', 'nothing', 'false', 'fake', 'untrue',
            'denied', 'rejected', 'opposed', 'disagreed', 'contradicts'
        }
        
        text1_words = set(text1.lower().split())
        text2_words = set(text2.lower().split())
        
        # Check for contradiction keywords
        contradiction_count = 0
        for keyword in contradiction_keywords:
            if keyword in text1_words or keyword in text2_words:
                contradiction_count += 1
        
        return min(contradiction_count * 0.1, 1.0)
    
    def _determine_verdict(self, support_score: float, refute_score: float, neutral_score: float) -> Tuple[str, float]:
        """Determine verdict and confidence based on scores."""
        max_score = max(support_score, refute_score, neutral_score)
        
        if support_score > refute_score and support_score > neutral_score:
            verdict = "SUPPORTED"
            confidence = support_score
        elif refute_score > support_score and refute_score > neutral_score:
            verdict = "REFUTED"
            confidence = refute_score
        else:
            verdict = "NOT_ENOUGH_INFO"
            confidence = neutral_score
        
        return verdict, confidence
    
    def _determine_final_verdict(self, support_score: float, refute_score: float, 
                                neutral_score: float, evidence_count: int) -> Tuple[str, float]:
        """Determine final verdict based on aggregated scores."""
        if evidence_count == 0:
            return "NOT_ENOUGH_INFO", 0.0
        
        # Adjust confidence based on evidence count
        evidence_factor = min(evidence_count / 5.0, 1.0)  # Cap at 5 evidence sources
        
        if support_score > refute_score + 0.1:  # Require clear margin
            verdict = "SUPPORTED"
            confidence = support_score * evidence_factor
        elif refute_score > support_score + 0.1:  # Require clear margin
            verdict = "REFUTED"
            confidence = refute_score * evidence_factor
        else:
            verdict = "NOT_ENOUGH_INFO"
            confidence = neutral_score * evidence_factor
        
        return verdict, confidence
    
    def _generate_reasoning(self, claim: str, evidence: str, support_score: float, 
                          refute_score: float, similarity_score: float) -> str:
        """Generate human-readable reasoning for the verification result."""
        reasoning_parts = []
        
        # Similarity analysis
        if similarity_score > 0.7:
            reasoning_parts.append("High semantic similarity between claim and evidence.")
        elif similarity_score > 0.4:
            reasoning_parts.append("Moderate semantic similarity between claim and evidence.")
        else:
            reasoning_parts.append("Low semantic similarity between claim and evidence.")
        
        # Support/refute analysis
        if support_score > 0.6:
            reasoning_parts.append("Evidence strongly supports the claim.")
        elif refute_score > 0.6:
            reasoning_parts.append("Evidence strongly contradicts the claim.")
        elif support_score > refute_score:
            reasoning_parts.append("Evidence provides some support for the claim.")
        elif refute_score > support_score:
            reasoning_parts.append("Evidence provides some contradiction to the claim.")
        else:
            reasoning_parts.append("Evidence is neutral regarding the claim.")
        
        return " ".join(reasoning_parts)

# Example usage
if __name__ == "__main__":
    verifier = CrossCheckVerifier()
    
    # Test claim and evidence
    claim = "The moon landing was faked by NASA in 1969."
    evidence = "NASA successfully landed astronauts on the moon in 1969 as part of the Apollo 11 mission."
    
    result = verifier.verify_claim_against_evidence(claim, evidence)
    
    print(f"Claim: {result.claim}")
    print(f"Evidence: {result.evidence_text}")
    print(f"Verdict: {result.overall_verdict}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Support Score: {result.entailment_score:.2f}")
    print(f"Refute Score: {result.contradiction_score:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    # Test cross-check analysis
    evidence_list = [
        {"content": evidence, "relevance_score": 0.8},
        {"content": "The Apollo 11 mission was a historic achievement in space exploration.", "relevance_score": 0.7},
        {"content": "Moon landing conspiracy theories have been debunked by scientific evidence.", "relevance_score": 0.9}
    ]
    
    analysis = verifier.cross_check_claim(claim, evidence_list)
    
    print(f"\nCross-Check Analysis:")
    print(f"Final Verdict: {analysis.final_verdict}")
    print(f"Confidence: {analysis.confidence:.2f}")
    print(f"Evidence Count: {analysis.evidence_count}")
    print(f"Reliable Evidence Count: {analysis.reliable_evidence_count}")
    print(f"Overall Support Score: {analysis.overall_support_score:.2f}")
    print(f"Overall Refute Score: {analysis.overall_refute_score:.2f}")
