"""
News Ingestion Module

This module handles the extraction and preprocessing of claims from various input formats
including text, images, and videos.
"""

print("before importing re")
import re
print("after importing re")

print("before importing logging")
import logging
print("after importing logging")

print("before importing typing")
from typing import Dict, List, Optional, Union, Tuple
print("after importing typing")

print("before importing dataclasses")
from dataclasses import dataclass
print("after importing dataclasses")

print("before importing PIL")
from PIL import Image
print("after importing PIL")

print("before importing cv2")
import cv2
print("after importing cv2")

print("before importing pytesseract")
import pytesseract
print("after importing pytesseract")

print("before importing spacy")
import spacy
print("after importing spacy")

print("before importing transformers")
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
print("after importing transformers")

print("before importing newspaper")
import newspaper
print("after importing newspaper")

print("before importing textblob")
from textblob import TextBlob
print("after importing textblob")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Claim:
    """Represents an extracted claim with metadata."""
    text: str
    confidence: float
    source_type: str  # 'text', 'image', 'video'
    entities: List[Dict]
    sentiment: Dict
    keywords: List[str]
    timestamp: Optional[str] = None
    location: Optional[str] = None

class NewsIngestion:
    """Handles ingestion and claim extraction from various media types."""
    
    def __init__(self):
        """Initialize the ingestion pipeline with required models."""
        print("before loading spacy model")
        self.nlp = spacy.load("en_core_web_sm")
        print("after loading spacy model")
        # Initialize claim detection model
        print("before initializing claim detection model")
        self.claim_detector = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            tokenizer="microsoft/DialoGPT-medium"
        )
        print("after initializing claim detection model")

        # Initialize summarization model for claim extraction
        print("before initializing summarization model")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn"
        )
        print("after initializing summarization model")
        logger.info("News ingestion pipeline initialized")
    
    def extract_claim_from_text(self, text: str) -> Claim:
        """Extract the main claim from text input."""
        try:
            # Clean and preprocess text
            cleaned_text = self._clean_text(text)
            
            # Extract entities
            doc = self.nlp(cleaned_text)
            entities = [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            # Analyze sentiment
            blob = TextBlob(cleaned_text)
            sentiment = {
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity
            }
            
            # Extract main claim using summarization
            if len(cleaned_text.split()) > 50:
                summary = self.summarizer(cleaned_text, max_length=100, min_length=30)
                main_claim = summary[0]['summary_text']
            else:
                main_claim = cleaned_text
            
            # Calculate confidence based on text characteristics
            confidence = self._calculate_text_confidence(cleaned_text)
            
            return Claim(
                text=main_claim,
                confidence=confidence,
                source_type="text",
                entities=entities,
                sentiment=sentiment,
                keywords=keywords
            )
            
        except Exception as e:
            logger.error(f"Error extracting claim from text: {e}")
            return Claim(
                text=text,
                confidence=0.0,
                source_type="text",
                entities=[],
                sentiment={"polarity": 0.0, "subjectivity": 0.0},
                keywords=[]
            )
    
    def extract_claim_from_image(self, image_path: str) -> Claim:
        """Extract text and claims from image using OCR."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert to PIL Image for OCR
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(pil_image)
            
            if not extracted_text.strip():
                logger.warning("No text found in image")
                return Claim(
                    text="",
                    confidence=0.0,
                    source_type="image",
                    entities=[],
                    sentiment={"polarity": 0.0, "subjectivity": 0.0},
                    keywords=[]
                )
            
            # Process extracted text as regular text claim
            claim = self.extract_claim_from_text(extracted_text)
            claim.source_type = "image"
            
            # Lower confidence for OCR-extracted text
            claim.confidence *= 0.8
            
            return claim
            
        except Exception as e:
            logger.error(f"Error extracting claim from image: {e}")
            return Claim(
                text="",
                confidence=0.0,
                source_type="image",
                entities=[],
                sentiment={"polarity": 0.0, "subjectivity": 0.0},
                keywords=[]
            )
    
    def extract_claim_from_url(self, url: str) -> Claim:
        """Extract claim from a news article URL."""
        import time
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            try:
                # Use newspaper3k to extract article content with user-agent header
                article = newspaper.Article(url)
                # Set custom user-agent to mimic browser
                article.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
                article.download()
                article.parse()

                # Check if we got meaningful content
                if not article.title and not article.text:
                    raise ValueError("No content extracted from URL")

                # Combine title and text for claim extraction
                full_text = f"{article.title}\n\n{article.text}"

                claim = self.extract_claim_from_text(full_text)
                claim.source_type = "url"
                claim.timestamp = article.publish_date.isoformat() if article.publish_date else None

                return claim

            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to extract content from URL {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.info(f"Using URL as fallback claim text: {url}")
                    return Claim(
                        text=url,  # Use URL itself as claim text
                        confidence=0.1,  # Low confidence for fallback
                        source_type="url",
                        entities=[],
                        sentiment={"polarity": 0.0, "subjectivity": 0.0},
                        keywords=[]
                    )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text input."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        doc = self.nlp(text)
        
        # Extract nouns, proper nouns, and adjectives
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                not token.is_punct and
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Remove duplicates and return top keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:10]  # Return top 10 keywords
    
    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for text-based claims."""
        confidence = 0.5  # Base confidence
        
        # Length factor
        word_count = len(text.split())
        if word_count > 10:
            confidence += 0.2
        if word_count > 50:
            confidence += 0.1
        
        # Sentence structure factor
        sentences = text.split('.')
        if len(sentences) > 1:
            confidence += 0.1
        
        # Entity presence factor
        doc = self.nlp(text)
        if len(doc.ents) > 0:
            confidence += 0.1
        
        # Cap confidence at 1.0
        return min(confidence, 1.0)
    
    def process_input(self, input_data: Union[str, Dict]) -> Claim:
        """Main entry point for processing various input types."""
        if isinstance(input_data, str):
            # Check if it's a URL
            if input_data.startswith(('http://', 'https://')):
                return self.extract_claim_from_url(input_data)
            else:
                return self.extract_claim_from_text(input_data)
        
        elif isinstance(input_data, dict):
            input_type = input_data.get('type', 'text')
            
            if input_type == 'text':
                return self.extract_claim_from_text(input_data['content'])
            elif input_type == 'image':
                return self.extract_claim_from_image(input_data['path'])
            elif input_type == 'url':
                return self.extract_claim_from_url(input_data['url'])
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
        
        else:
            raise ValueError("Input must be a string or dictionary")

# Example usage
if __name__ == "__main__":
    ingestion = NewsIngestion()
    
    # Test with text input
    text_claim = "The moon landing was faked by NASA in 1969."
    claim = ingestion.process_input(text_claim)
    print(f"Extracted claim: {claim.text}")
    print(f"Confidence: {claim.confidence}")
    print(f"Keywords: {claim.keywords}")
    print(f"Entities: {claim.entities}")
