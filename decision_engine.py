"""
Final Decision Engine Module

This module combines all verification scores and evidence to make a final
decision about the authenticity of a claim.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VerificationScores:
    """Represents all verification scores."""
    claim_confidence: float
    source_reliability_score: float
    cross_check_support_score: float
    cross_check_refute_score: float
    cross_check_neutral_score: float
    multimodal_authenticity_score: float
    evidence_count: int
    reliable_evidence_count: int

@dataclass
class FinalDecision:
    """Represents the final decision about a claim."""
    claim: str
    verdict: str  # 'LIKELY_REAL', 'LIKELY_FAKE', 'NEEDS_VERIFICATION'
    confidence: float
    overall_score: float
    verification_scores: VerificationScores
    reasoning: str
    evidence_summary: Dict
    warnings: List[str]
    recommendations: List[str]
    timestamp: str

class DecisionEngine:
    """Combines all verification results to make final decisions."""
    
    def __init__(self):
        """Initialize the decision engine with scoring weights."""
        # Define weights for different verification components
        self.weights = {
            'claim_confidence': 0.1,
            'source_reliability': 0.2,
            'cross_check_support': 0.3,
            'cross_check_refute': 0.3,
            'multimodal_authenticity': 0.1
        }
        
        # Thresholds for decision making
        self.thresholds = {
            'likely_real_min_score': 0.7,
            'likely_fake_max_score': 0.4,
            'min_evidence_count': 3,  # Require at least 3 quality articles
            'min_reliable_evidence': 1,
            'high_confidence_threshold': 0.8,
            'low_confidence_threshold': 0.3,
            'controversial_consensus_threshold': 0.6  # Higher threshold for controversial claims
        }
        
        logger.info("Decision engine initialized")
    
    def make_final_decision(self, 
                          claim: str,
                          claim_confidence: float,
                          source_analysis: Dict,
                          cross_check_analysis: Dict,
                          multimodal_analysis: Optional[Dict] = None,
                          evidence_summary: Dict = None) -> FinalDecision:
        """Make final decision based on all verification results."""
        try:
            print(f"DEBUG: Starting final decision making for claim: {claim[:120]}")
            
            # Extract scores from analyses
            verification_scores = self._extract_verification_scores(
                claim_confidence, source_analysis, cross_check_analysis, multimodal_analysis
            )
            print(f"DEBUG: Verification scores extracted: claim_confidence={verification_scores.claim_confidence}, source_reliability={verification_scores.source_reliability_score}, cross_check_support={verification_scores.cross_check_support_score}, evidence_count={verification_scores.evidence_count}")
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(verification_scores)
            print(f"DEBUG: Overall score calculated: {overall_score}")
            
            # Determine verdict
            verdict, confidence = self._determine_verdict(verification_scores, overall_score)
            print(f"DEBUG: Verdict determined: {verdict}, confidence: {confidence}")
            
            # Generate reasoning
            reasoning = self._generate_reasoning(verification_scores, verdict, overall_score)
            print(f"DEBUG: Reasoning generated: {reasoning}")
            
            # Generate warnings and recommendations
            warnings, recommendations = self._generate_warnings_and_recommendations(
                verification_scores, verdict, confidence
            )
            
            # Prepare evidence summary
            if evidence_summary is None:
                evidence_summary = self._prepare_evidence_summary(
                    source_analysis, cross_check_analysis, multimodal_analysis
                )
            
            print(f"DEBUG: Final decision completed: verdict={verdict}, overall_score={overall_score}, warnings_count={len(warnings)}")
            
            return FinalDecision(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                overall_score=overall_score,
                verification_scores=verification_scores,
                reasoning=reasoning,
                evidence_summary=evidence_summary,
                warnings=warnings,
                recommendations=recommendations,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return self._create_error_decision(claim, str(e))
    
    def _extract_verification_scores(self, 
                                   claim_confidence: float,
                                   source_analysis: Dict,
                                   cross_check_analysis: Dict,
                                   multimodal_analysis: Optional[Dict]) -> VerificationScores:
        """Extract verification scores from all analyses."""
        # Source reliability score
        source_reliability_score = source_analysis.get('credibility_score', 0.5)
        
        # Cross-check scores
        cross_check_support_score = cross_check_analysis.get('overall_support_score', 0.0)
        cross_check_refute_score = cross_check_analysis.get('overall_refute_score', 0.0)
        cross_check_neutral_score = cross_check_analysis.get('overall_neutral_score', 0.0)
        
        # Evidence counts
        evidence_count = cross_check_analysis.get('evidence_count', 0)
        reliable_evidence_count = cross_check_analysis.get('reliable_evidence_count', 0)
        
        # Multimodal authenticity score
        multimodal_authenticity_score = 0.5  # Default neutral score
        if multimodal_analysis:
            multimodal_authenticity_score = multimodal_analysis.get('overall_authenticity_score', 0.5)
        
        return VerificationScores(
            claim_confidence=claim_confidence,
            source_reliability_score=source_reliability_score,
            cross_check_support_score=cross_check_support_score,
            cross_check_refute_score=cross_check_refute_score,
            cross_check_neutral_score=cross_check_neutral_score,
            multimodal_authenticity_score=multimodal_authenticity_score,
            evidence_count=evidence_count,
            reliable_evidence_count=reliable_evidence_count
        )
    
    def _calculate_overall_score(self, scores: VerificationScores) -> float:
        """Calculate overall score using advanced consensus-based aggregation."""
        # Use consensus-based aggregation instead of simple weighted averaging
        consensus_score = self._calculate_consensus_score(scores)

        # Adjust for evidence quality and other factors
        evidence_factor = self._calculate_evidence_factor(scores)
        final_score = consensus_score * evidence_factor

        # Additional adjustments for multimodal and claim confidence
        if scores.multimodal_authenticity_score < 0.4:
            final_score *= 0.9  # Slight penalty for suspicious media

        if scores.claim_confidence > 0.8:
            final_score *= 1.05  # Bonus for high claim confidence

        return max(0.0, min(1.0, final_score))

    def _calculate_consensus_score(self, scores: VerificationScores) -> float:
        """Calculate consensus score using advanced aggregation method."""
        # For now, use the existing cross-check scores as proxy for article-level analysis
        # In a full implementation, this would analyze individual articles

        # Calculate net support/refute strength (-1 to +1 range)
        net_evidence = scores.cross_check_support_score - scores.cross_check_refute_score

        # Apply credibility weighting
        credibility_weight = scores.source_reliability_score

        # Calculate consensus score
        # consensus_score = (sum(supporting_weights) - sum(contradicting_weights)) / total_weighted_articles
        consensus_score = net_evidence * credibility_weight

        # Normalize to 0-1 range (reality score interpretation: +1 = supports true, 0 = neutral, -1 = indicates false)
        # Convert from (-1, +1) range to (0, 1) range
        normalized_score = (consensus_score + 1.0) / 2.0

        return max(0.0, min(1.0, normalized_score))
    
    def _calculate_evidence_factor(self, scores: VerificationScores) -> float:
        """Calculate evidence quality factor."""
        factor = 1.0
        
        # Penalize insufficient evidence
        if scores.evidence_count < self.thresholds['min_evidence_count']:
            factor *= 0.7
        
        if scores.reliable_evidence_count < self.thresholds['min_reliable_evidence']:
            factor *= 0.4
        
        # Bonus for high evidence count
        if scores.evidence_count >= 5:
            factor *= 1.1
        
        if scores.reliable_evidence_count >= 3:
            factor *= 1.1
        
        return min(factor, 1.2)  # Cap bonus at 20%
    
    def _determine_verdict(self, scores: VerificationScores, overall_score: float) -> Tuple[str, float]:
        """Determine final verdict and confidence with minimum evidence requirements."""
        # Check minimum evidence requirements first
        if scores.evidence_count < self.thresholds['min_evidence_count']:
            verdict = "NEEDS_VERIFICATION"
            confidence = 0.1  # Very low confidence due to insufficient evidence
            return verdict, confidence

        if scores.reliable_evidence_count < self.thresholds['min_reliable_evidence']:
            verdict = "NEEDS_VERIFICATION"
            confidence = 0.2  # Low confidence due to lack of reliable sources
            return verdict, confidence

        # Check for controversial claims (high disagreement)
        disagreement_ratio = 0.0
        if scores.cross_check_support_score + scores.cross_check_refute_score > 0:
            disagreement_ratio = abs(scores.cross_check_support_score - scores.cross_check_refute_score) / (scores.cross_check_support_score + scores.cross_check_refute_score)

        # For controversial claims, require higher consensus threshold
        if disagreement_ratio > 0.4:  # Significant disagreement
            real_threshold = self.thresholds['controversial_consensus_threshold']
            fake_threshold = 1.0 - self.thresholds['controversial_consensus_threshold']
        else:
            real_threshold = self.thresholds['likely_real_min_score']
            fake_threshold = self.thresholds['likely_fake_max_score']

        # Determine verdict based on adjusted thresholds
        if overall_score >= real_threshold:
            verdict = "LIKELY_REAL"
        elif overall_score <= fake_threshold:
            verdict = "LIKELY_FAKE"
        else:
            verdict = "NEEDS_VERIFICATION"

        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(scores, overall_score, verdict)

        return verdict, confidence
    
    def _calculate_confidence(self, scores: VerificationScores, overall_score: float, verdict: str) -> float:
        """Calculate confidence using advanced method based on evidence strength and disagreement."""
        # Base confidence on overall score
        base_confidence = overall_score

        # Calculate strongest evidence weight
        strongest_evidence = max(scores.cross_check_support_score, scores.cross_check_refute_score, scores.cross_check_neutral_score)
        total_weight = scores.cross_check_support_score + scores.cross_check_refute_score + scores.cross_check_neutral_score

        # Avoid division by zero
        if total_weight > 0:
            evidence_strength_ratio = strongest_evidence / total_weight
        else:
            evidence_strength_ratio = 0.5

        # Calculate disagreement ratio (penalize conflicting evidence)
        # If 40%+ articles contradict majority, reduce confidence
        support_refute_diff = abs(scores.cross_check_support_score - scores.cross_check_refute_score)
        total_evidence_strength = scores.cross_check_support_score + scores.cross_check_refute_score

        if total_evidence_strength > 0:
            disagreement_ratio = min(support_refute_diff / total_evidence_strength, 1.0)
        else:
            disagreement_ratio = 0.0

        # Apply disagreement penalty
        disagreement_penalty = disagreement_ratio * 0.3  # Max 30% penalty

        # Calculate final confidence
        confidence = min(1.0, evidence_strength_ratio) * (1 - disagreement_penalty)

        # Additional adjustments for evidence quality
        if scores.evidence_count > 5:
            confidence += 0.1
        elif scores.evidence_count < 2:
            confidence -= 0.2

        if scores.reliable_evidence_count > 3:
            confidence += 0.1
        elif scores.reliable_evidence_count == 0:
            confidence -= 0.7

        # Adjust confidence based on source reliability
        if scores.source_reliability_score > 0.8:
            confidence += 0.1
        elif scores.source_reliability_score < 0.3:
            confidence -= 0.1

        # Cap confidence
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, scores: VerificationScores, verdict: str, overall_score: float) -> str:
        """Generate human-readable reasoning for the decision."""
        reasoning_parts = []
        
        # Overall score explanation
        if overall_score >= 0.7:
            reasoning_parts.append("The evidence strongly supports the claim.")
        elif overall_score <= 0.3:
            reasoning_parts.append("The evidence strongly contradicts the claim.")
        else:
            reasoning_parts.append("The evidence is mixed or insufficient.")
        
        # Source reliability
        if scores.source_reliability_score >= 0.7:
            reasoning_parts.append("Sources are generally reliable.")
        elif scores.source_reliability_score <= 0.3:
            reasoning_parts.append("Sources have questionable reliability.")
        
        # Evidence quantity
        if scores.evidence_count >= 5:
            reasoning_parts.append("Multiple sources of evidence were found.")
        elif scores.evidence_count < 2:
            reasoning_parts.append("Limited evidence was available.")
        
        # Cross-check results
        if scores.cross_check_support_score > scores.cross_check_refute_score + 0.2:
            reasoning_parts.append("Cross-checking supports the claim.")
        elif scores.cross_check_refute_score > scores.cross_check_support_score + 0.2:
            reasoning_parts.append("Cross-checking contradicts the claim.")
        else:
            reasoning_parts.append("Cross-checking results are inconclusive.")
        
        return " ".join(reasoning_parts)
    
    def _generate_warnings_and_recommendations(self, 
                                            scores: VerificationScores, 
                                            verdict: str, 
                                            confidence: float) -> Tuple[List[str], List[str]]:
        """Generate warnings and recommendations."""
        warnings = []
        recommendations = []
        
        # Low confidence warnings
        if confidence < self.thresholds['low_confidence_threshold']:
            warnings.append("Low confidence in verification result")
            recommendations.append("Manual review recommended")
        
        # Insufficient evidence warnings
        if scores.evidence_count < self.thresholds['min_evidence_count']:
            warnings.append("Insufficient evidence found")
            recommendations.append("Search for additional sources")
        
        if scores.reliable_evidence_count < self.thresholds['min_reliable_evidence']:
            warnings.append("Limited reliable evidence")
            recommendations.append("Verify source credibility")
        
        # Source reliability warnings
        if scores.source_reliability_score < 0.4:
            warnings.append("Questionable source reliability")
            recommendations.append("Cross-reference with trusted sources")
        
        # Cross-check warnings
        if abs(scores.cross_check_support_score - scores.cross_check_refute_score) < 0.1:
            warnings.append("Conflicting evidence found")
            recommendations.append("Further investigation needed")
        
        # High confidence recommendations
        if confidence >= self.thresholds['high_confidence_threshold']:
            recommendations.append("High confidence result - suitable for sharing")
        
        return warnings, recommendations
    
    def _prepare_evidence_summary(self, 
                               source_analysis: Dict,
                               cross_check_analysis: Dict,
                               multimodal_analysis: Optional[Dict]) -> Dict:
        """Prepare evidence summary for the decision."""
        summary = {
            'total_evidence_sources': cross_check_analysis.get('evidence_count', 0),
            'reliable_sources': cross_check_analysis.get('reliable_evidence_count', 0),
            'source_reliability': source_analysis.get('credibility_score', 0.5),
            'cross_check_verdict': cross_check_analysis.get('final_verdict', 'UNKNOWN'),
            'multimodal_status': 'N/A'
        }
        
        if multimodal_analysis:
            summary['multimodal_status'] = multimodal_analysis.get('verification_status', 'UNKNOWN')
            summary['multimodal_authenticity'] = multimodal_analysis.get('overall_authenticity_score', 0.5)
        
        return summary
    
    def _create_error_decision(self, claim: str, error_message: str) -> FinalDecision:
        """Create error decision when processing fails."""
        return FinalDecision(
            claim=claim,
            verdict="NEEDS_VERIFICATION",
            confidence=0.0,
            overall_score=0.5,
            verification_scores=VerificationScores(
                claim_confidence=0.0,
                source_reliability_score=0.5,
                cross_check_support_score=0.0,
                cross_check_refute_score=0.0,
                cross_check_neutral_score=1.0,
                multimodal_authenticity_score=0.5,
                evidence_count=0,
                reliable_evidence_count=0
            ),
            reasoning=f"Error in verification process: {error_message}",
            evidence_summary={'error': error_message},
            warnings=[f"Processing error: {error_message}"],
            recommendations=["Manual review required due to processing error"],
            timestamp=datetime.now().isoformat()
        )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update scoring weights."""
        self.weights.update(new_weights)
        logger.info(f"Updated decision engine weights: {self.weights}")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update decision thresholds."""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated decision engine thresholds: {self.thresholds}")
    
    def get_decision_statistics(self) -> Dict:
        """Get statistics about decision making."""
        return {
            'weights': self.weights,
            'thresholds': self.thresholds,
            'version': '1.0.0'
        }

# Example usage
if __name__ == "__main__":
    decision_engine = DecisionEngine()
    
    # Example verification results
    claim = "The moon landing was faked by NASA in 1969."
    claim_confidence = 0.8
    
    source_analysis = {
        'credibility_score': 0.3,
        'category': 'unreliable',
        'domain': 'conspiracy-site.com'
    }
    
    cross_check_analysis = {
        'overall_support_score': 0.1,
        'overall_refute_score': 0.8,
        'overall_neutral_score': 0.1,
        'final_verdict': 'REFUTED',
        'evidence_count': 5,
        'reliable_evidence_count': 4
    }
    
    multimodal_analysis = {
        'overall_authenticity_score': 0.7,
        'verification_status': 'AUTHENTIC'
    }
    
    # Make final decision
    decision = decision_engine.make_final_decision(
        claim=claim,
        claim_confidence=claim_confidence,
        source_analysis=source_analysis,
        cross_check_analysis=cross_check_analysis,
        multimodal_analysis=multimodal_analysis
    )
    
    print(f"Claim: {decision.claim}")
    print(f"Verdict: {decision.verdict}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Overall Score: {decision.overall_score:.2f}")
    print(f"Reasoning: {decision.reasoning}")
    
    if decision.warnings:
        print(f"Warnings: {', '.join(decision.warnings)}")
    
    if decision.recommendations:
        print(f"Recommendations: {', '.join(decision.recommendations)}")
    
    print(f"\nEvidence Summary:")
    for key, value in decision.evidence_summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nVerification Scores:")
    if is_dataclass(decision.verification_scores):
        scores_dict = asdict(decision.verification_scores)
    else:
        scores_dict = decision.verification_scores  # fallback if already dict
    for key, value in scores_dict.items():
        print(f"  {key}: {value}")
