"""
Multi-Modal Verification Module

This module handles verification of images and videos using reverse image search,
deepfake detection, and other multi-modal analysis techniques.
"""

import os
import logging
import aiohttp
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import hashlib
import base64
from io import BytesIO
import json
from datetime import datetime
import pytesseract
from transformers import pipeline
import torch
from dotenv import load_dotenv
        
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import hashlib
import base64
from io import BytesIO
import json
from datetime import datetime
import pytesseract
from transformers import pipeline
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # This will load the .env file


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Tesseract
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\naroy\tesseract.exe'
    logger.info("Tesseract configured successfully")
except Exception as e:
    logger.warning(f"Could not configure Tesseract: {e}. OCR functionality will be limited")

@dataclass
class ImageAnalysis:
    """Represents analysis results for an image."""
    image_path: str
    image_hash: str
    reverse_search_results: List[Dict]
    ocr_text: str
    metadata: Dict
    tampering_indicators: List[str]
    authenticity_score: float
    confidence: float

@dataclass
class VideoAnalysis:
    """Represents analysis results for a video."""
    video_path: str
    video_hash: str
    keyframes: List[str]
    reverse_search_results: List[Dict]
    deepfake_indicators: List[str]
    audio_analysis: Dict
    authenticity_score: float
    confidence: float

@dataclass
class MultiModalVerification:
    """Represents comprehensive multi-modal verification results."""
    media_type: str  # 'image', 'video', 'text'
    analysis_results: Dict
    overall_authenticity_score: float
    verification_status: str  # 'AUTHENTIC', 'SUSPICIOUS', 'FAKE', 'UNKNOWN'
    confidence: float
    warnings: List[str]
    recommendations: List[str]

class MultiModalVerifier:
    """Handles multi-modal verification of images and videos."""
    
    def __init__(self):
        """Initialize the multi-modal verifier."""
        self.clarifai_api_key = os.getenv('CLARIFAI_API_KEY')
        self.clarifai_user_id = os.getenv('CLARIFAI_USER_ID')
        self.clarifai_app_id = os.getenv('CLARIFAI_APP_ID')
        self.tineye_api_key = os.getenv('TINEYE_API_KEY')
        self.google_vision_api_key = os.getenv('GOOGLE_VISION_API_KEY')
        
        # Log API key status
        if not all([self.clarifai_api_key, self.clarifai_user_id, self.clarifai_app_id]):
            logger.warning("Clarifai configuration incomplete. Need API key, user ID, and app ID.")
            if not self.clarifai_api_key:
                logger.warning("Missing Clarifai API key")
            if not self.clarifai_user_id:
                logger.warning("Missing Clarifai user ID")
            if not self.clarifai_app_id:
                logger.warning("Missing Clarifai app ID")
        else:
            logger.info("Clarifai API configured successfully")
            
        if not self.tineye_api_key:
            logger.warning("TinEye API key not found. Reverse image search with TinEye will be disabled.")
        else:
            logger.info("TinEye API key configured successfully")
            
        if not self.google_vision_api_key:
            logger.warning("Google Vision API key not found. Reverse image search with Google Vision will be disabled.")
        else:
            logger.info("Google Vision API key configured successfully")
        
        # Initialize deepfake detection model using specialized model
        try:
            self.deepfake_detector = pipeline(
                "image-classification",
                model="DarkVision/Deepfake_detection_image",  # Specialized model for deepfake detection
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Loaded specialized deepfake detection model")
        except Exception as e:
            try:
                # Fallback to a simpler but still effective model
                self.deepfake_detector = pipeline(
                    "image-classification",
                    model="drew7557/deepfake-detection-resnet",  # Alternative deepfake detection model
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Loaded fallback deepfake detection model")
            except Exception as e2:
                logger.warning(f"Could not load deepfake detection models: {e2}")
                self.deepfake_detector = None
        
        # Initialize image tampering detection
        self.tampering_detector = self._initialize_tampering_detector()
        
        logger.info("Multi-modal verifier initialized")
    
    async def verify_image(self, image_path: str) -> ImageAnalysis:
        """Perform comprehensive verification of an image."""
        try:
            # Load and analyze image
            image = Image.open(image_path)
            image_hash = self._calculate_image_hash(image)
            
            # Extract metadata
            metadata = self._extract_image_metadata(image)
            
            # Perform OCR
            ocr_text = self._extract_text_from_image(image)
            
            # Reverse image search
            reverse_search_results = await self._reverse_image_search(image)
            
            # Detect tampering indicators
            tampering_indicators = self._detect_tampering_indicators(image_path)
            
            # Calculate authenticity score
            authenticity_score = self._calculate_image_authenticity_score(
                reverse_search_results, tampering_indicators, metadata
            )
            
            # Calculate confidence
            confidence = self._calculate_image_confidence(
                reverse_search_results, tampering_indicators, ocr_text
            )
            
            return ImageAnalysis(
                image_path=image_path,
                image_hash=image_hash,
                reverse_search_results=reverse_search_results,
                ocr_text=ocr_text,
                metadata=metadata,
                tampering_indicators=tampering_indicators,
                authenticity_score=authenticity_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error verifying image: {e}")
            return ImageAnalysis(
                image_path=image_path,
                image_hash="",
                reverse_search_results=[],
                ocr_text="",
                metadata={},
                tampering_indicators=[],
                authenticity_score=0.0,
                confidence=0.0
            )
    
    async def verify_video(self, video_path: str) -> VideoAnalysis:
        """Perform comprehensive verification of a video."""
        try:
            # Calculate video hash
            video_hash = self._calculate_video_hash(video_path)
            
            # Extract keyframes
            keyframes = self._extract_keyframes(video_path)
            
            # Reverse search keyframes
            reverse_search_results = []
            for keyframe_path in keyframes:
                keyframe_image = Image.open(keyframe_path)
                results = await self._reverse_image_search(keyframe_image)
                reverse_search_results.extend(results)
            
            # Detect deepfake indicators
            deepfake_indicators = self._detect_deepfake_indicators(video_path)
            
            # Analyze audio
            audio_analysis = self._analyze_audio(video_path)
            
            # Calculate authenticity score
            authenticity_score = self._calculate_video_authenticity_score(
                reverse_search_results, deepfake_indicators, audio_analysis
            )
            
            # Calculate confidence
            confidence = self._calculate_video_confidence(
                reverse_search_results, deepfake_indicators, len(keyframes)
            )
            
            return VideoAnalysis(
                video_path=video_path,
                video_hash=video_hash,
                keyframes=keyframes,
                reverse_search_results=reverse_search_results,
                deepfake_indicators=deepfake_indicators,
                audio_analysis=audio_analysis,
                authenticity_score=authenticity_score,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error verifying video: {e}")
            return VideoAnalysis(
                video_path=video_path,
                video_hash="",
                keyframes=[],
                reverse_search_results=[],
                deepfake_indicators=[],
                audio_analysis={},
                authenticity_score=0.0,
                confidence=0.0
            )
    
    async def verify_multimodal_content(self, content_path: str, content_type: str) -> MultiModalVerification:
        """Main entry point for multi-modal verification."""
        try:
            warnings = []
            recommendations = []
            
            if content_type.lower() == 'image':
                analysis = await self.verify_image(content_path)
                verification_status = self._determine_image_verification_status(analysis)
                
                # Check for AI generation
                if analysis.metadata.get('ai_indicators', []):
                    warnings.extend(analysis.metadata['ai_indicators'])
                    recommendations.append("This appears to be an AI-generated image, not a real photograph")
                    verification_status = "SYNTHETIC"  # New status for AI-generated content
                
                if analysis.authenticity_score < 0.3:
                    warnings.append("Low authenticity score detected")
                    recommendations.append("Consider additional verification")
                
                if analysis.tampering_indicators:
                    warnings.append("Potential tampering indicators found")
                    recommendations.append("Manual review recommended")
                
            elif content_type.lower() == 'video':
                analysis = self.verify_video(content_path)
                verification_status = self._determine_video_verification_status(analysis)
                
                if analysis.authenticity_score < 0.3:
                    warnings.append("Low authenticity score detected")
                    recommendations.append("Consider additional verification")
                
                if analysis.deepfake_indicators:
                    warnings.append("Potential deepfake indicators found")
                    recommendations.append("Manual review recommended")
                
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            return MultiModalVerification(
                media_type=content_type,
                analysis_results=analysis.__dict__,
                overall_authenticity_score=analysis.authenticity_score,
                verification_status=verification_status,
                confidence=analysis.confidence,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in multi-modal verification: {e}")
            return MultiModalVerification(
                media_type=content_type,
                analysis_results={},
                overall_authenticity_score=0.0,
                verification_status="UNKNOWN",
                confidence=0.0,
                warnings=[f"Error: {str(e)}"],
                recommendations=["Manual review required"]
            )
    
    def _calculate_image_hash(self, image: Image.Image) -> str:
        """Calculate perceptual hash of an image."""
        try:
            # Convert to grayscale and resize for consistent hashing
            image_gray = image.convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            
            # Calculate hash
            pixels = list(image_gray.getdata())
            avg = sum(pixels) / len(pixels)
            hash_bits = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
            
            # Convert to hex
            hash_hex = hex(int(hash_bits, 2))[2:].zfill(16)
            return hash_hex
            
        except Exception as e:
            logger.error(f"Error calculating image hash: {e}")
            return ""
    
    def _calculate_video_hash(self, video_path: str) -> str:
        """Calculate hash of a video file."""
        try:
            hash_md5 = hashlib.md5()
            with open(video_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating video hash: {e}")
            return ""
    
    def _extract_image_metadata(self, image: Image.Image) -> Dict:
        """Extract metadata from image."""
        metadata = {}
        
        try:
            # Basic image properties
            metadata['size'] = image.size
            metadata['mode'] = image.mode
            metadata['format'] = image.format
            
            # EXIF data
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
                metadata['exif'] = exif_data
            
            # Check for common tampering indicators and AI generation in metadata
            tampering_flags = []
            ai_indicators = []
            
            if 'exif' in metadata:
                exif = metadata['exif']
                # Check for image editing software
                if 271 in exif and 'Photoshop' in str(exif[271]):
                    tampering_flags.append('Photoshop detected')
                if 272 in exif and 'Adobe' in str(exif[272]):
                    tampering_flags.append('Adobe software detected')
                    
                # Check for AI generation markers
                if 271 in exif:
                    software = str(exif[271]).lower()
                    ai_markers = ['ideogram', 'midjourney', 'dall-e', 'stable diffusion']
                    for marker in ai_markers:
                        if marker in software:
                            ai_indicators.append(f'Generated by {exif[271]}')
            
            metadata['tampering_flags'] = tampering_flags
            metadata['ai_indicators'] = ai_indicators
            
        except Exception as e:
            logger.error(f"Error extracting image metadata: {e}")
        
        return metadata
    
    def _preprocess_image_for_ocr(self, image: Image.Image) -> np.ndarray:
        """Preprocess image to improve OCR accuracy."""
        # Convert PIL Image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize image if too small (minimum 2000px width while maintaining aspect ratio)
        min_width = 2000
        if img_cv.shape[1] < min_width:
            scale = min_width / img_cv.shape[1]
            new_width = min_width
            new_height = int(img_cv.shape[0] * scale)
            img_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Deskew if needed
        coords = np.column_stack(np.where(denoised > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(
                denoised, M, (w, h), 
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
        
        return denoised

    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR with enhanced preprocessing."""
        try:
            # Check if Tesseract is properly configured
            if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                logger.warning("Tesseract not found at configured path. Please install Tesseract-OCR.")
                return "OCR unavailable - Tesseract not installed"
            
            # Convert image to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            processed_img = self._preprocess_image_for_ocr(image)
            
            # Configure Tesseract parameters for better accuracy and safer parsing
            custom_config = '--oem 3 --psm 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.:,!?()-_+=/@#$%& -c tessedit_pageseg_mode=1 -c tessedit_do_invert=0 --dpi 300'
            
            # Try multiple preprocessing variations with better error handling
            texts = []
            
            try:
                # Original preprocessed image
                text1 = pytesseract.image_to_string(processed_img, config=custom_config)
                if text1 and text1.strip():
                    # Clean the text to prevent quotation errors
                    cleaned_text = ' '.join(text1.strip().split())
                    texts.append(cleaned_text)
            except Exception as e:
                logger.warning(f"First OCR attempt failed: {e}")
            
            try:
                # Try with different threshold
                _, binary = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text2 = pytesseract.image_to_string(binary, config=custom_config)
                if text2 and text2.strip():
                    # Clean the text to prevent quotation errors
                    cleaned_text = ' '.join(text2.strip().split())
                    texts.append(cleaned_text)
            except Exception as e:
                logger.warning(f"Second OCR attempt failed: {e}")
            
            # Combine results and remove duplicates
            if texts:
                # Use the longest result as it's likely the most complete
                result = max(texts, key=len)
                return result
            
            return "No text detected in image"
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return f"OCR failed: {str(e)}"
    
    async def _reverse_image_search(self, image: Image.Image) -> List[Dict]:
        """Perform reverse image search using available APIs."""
        results = []
        
        try:
            # Convert image to base64 for API calls
            buffer = BytesIO()
            image.save(buffer, format='JPEG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Clarifai API (if available)
            if self.clarifai_api_key:
                try:
                    clarifai_results = await self._search_clarifai(image_base64)
                    if clarifai_results:
                        results.extend(clarifai_results)
                except Exception as e:
                    logger.error(f"Error in Clarifai search: {e}")
            
            # TinEye API (if available and no results yet)
            if self.tineye_api_key and not results:
                try:
                    tineye_results = await self._search_tineye(image_base64)
                    results.extend(tineye_results)
                except Exception as e:
                    logger.error(f"Error in TinEye search: {e}")
            
            # Google Vision API (if available and no results yet)
            if self.google_vision_api_key and not results:
                try:
                    google_results = await self._search_google_vision(image_base64)
                    results.extend(google_results)
                except Exception as e:
                    logger.error(f"Error in Google Vision search: {e}")
            
        except Exception as e:
            logger.error(f"Error in reverse image search: {e}")
        
        return results
    
    async def _search_clarifai(self, image_base64: str) -> List[Dict]:
        """Search using Clarifai's visual search API."""
        results = []
        try:
            headers = {
                'Authorization': f'Bearer {self.clarifai_api_key}',  # Using PAT token
                'Content-Type': 'application/json'
            }
            
            # Prepare the request body for Clarifai
            # Using Clarifai's Visual Search API v2
            data = {
                "user_app_id": {
                    "user_id": self.clarifai_user_id,
                    "app_id": self.clarifai_app_id
                },
                "inputs": [{
                    "data": {
                        "image": {
                            "base64": image_base64
                        }
                    }
                }]
            }
            
            # First, ensure the image isn't too large
            img_size = len(image_base64)
            if img_size > 1024 * 1024:  # If larger than 1MB
                # Compress the image
                image_data = base64.b64decode(image_base64)
                img = Image.open(BytesIO(image_data))
                
                # Calculate new size while maintaining aspect ratio
                ratio = min(1024/float(img.size[0]), 1024/float(img.size[1]))
                new_size = tuple([int(x*ratio) for x in img.size])
                
                # Resize and compress
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                data["inputs"][0]["data"]["image"]["base64"] = image_base64

            async with aiohttp.ClientSession() as session:
                # Use the visual search API endpoint
                api_url = f'https://api.clarifai.com/v2/users/{self.clarifai_user_id}/apps/{self.clarifai_app_id}/searches'
                
                # Modify the request body for the search endpoint
                search_data = {
                    "query": {
                        "ands": [{
                            "output": {
                                "input": {
                                    "data": {
                                        "image": {
                                            "base64": image_base64
                                        }
                                    }
                                }
                            }
                        }]
                    }
                }
                
                async with session.post(api_url, headers=headers, json=search_data) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        logger.warning(f"Clarifai API returned status {response.status}: {response_text}")
                        return results
                        
                    try:
                        result = json.loads(response_text)
                        if 'hits' in result:
                            for hit in result['hits']:
                                result_dict = {
                                    'source': 'Clarifai',
                                    'similarity': hit.get('score', 0),
                                    'metadata': {}
                                }
                                
                                # Extract image URL from the hit
                                if 'input' in hit and 'data' in hit['input']:
                                    result_dict['url'] = hit['input']['data'].get('image', {}).get('url', '')
                                
                                # Add metadata if available
                                if 'metadata' in hit:
                                    result_dict['metadata'] = hit['metadata']
                                
                                # Add the result if we have a URL
                                if result_dict['url']:
                                    results.append(result_dict)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Clarifai response: {e}")
                        
            return results
            
        except Exception as e:
            logger.error(f"Error in Clarifai search: {e}")
            return []

    async def _search_tineye(self, image_base64: str) -> List[Dict]:
        """Search using TinEye API."""
        try:
            headers = {
                'Content-Type': 'application/json',
                'X-API-Key': self.tineye_api_key
            }
            
            data = {
                'image': image_base64,
                'limit': 10
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.tineye.com/rest/search/',
                    headers=headers,
                    json=data
                ) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        result = await response.json()
                        matches = result.get('results', [])
                        logger.info(f"TinEye API found {len(matches)} matching images")
                        return matches
                    else:
                        logger.error(f"TinEye API error: Status {response.status} - {response_text}")
                        return []
        
        except Exception as e:
            logger.error(f"Error in TinEye search: {e}")
        
        return []
    
    async def _search_google_vision(self, image_base64: str) -> List[Dict]:
        """Search using Google Vision API."""
        try:
            url = f'https://vision.googleapis.com/v1/images:annotate?key={self.google_vision_api_key}'
            
            data = {
                'requests': [{
                    'image': {'content': image_base64},
                    'features': [{'type': 'WEB_DETECTION'}]
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        result = await response.json()
                        web_detection = result.get('responses', [{}])[0].get('webDetection', {})
                        matches = web_detection.get('pagesWithMatchingImages', [])
                        logger.info(f"Google Vision API found {len(matches)} matching images")
                        return matches
                    else:
                        logger.error(f"Google Vision API error: Status {response.status} - {response_text}")
                        return []
        
        except Exception as e:
            logger.error(f"Error in Google Vision search: {e}")
        
        return []
    
    def _detect_tampering_indicators(self, image_path: str) -> List[str]:
        """Detect potential tampering indicators in image."""
        indicators = []
        
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                return indicators
            
            # Check for copy-move forgery using block matching
            copy_move_score = self._detect_copy_move_forgery(image)
            if copy_move_score > 0.7:
                indicators.append('Potential copy-move forgery detected')
            
            # Check for splicing using edge detection
            splicing_score = self._detect_splicing(image)
            if splicing_score > 0.6:
                indicators.append('Potential image splicing detected')
            
            # Check for compression artifacts
            compression_score = self._analyze_compression_artifacts(image)
            if compression_score > 0.8:
                indicators.append('Unusual compression artifacts detected')
            
        except Exception as e:
            logger.error(f"Error detecting tampering indicators: {e}")
        
        return indicators
    
    def _detect_copy_move_forgery(self, image: np.ndarray) -> float:
        """Detect copy-move forgery using block matching."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Divide image into blocks
            block_size = 32
            h, w = gray.shape
            blocks = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    blocks.append(block)
            
            # Compare blocks for similarity
            similar_blocks = 0
            total_comparisons = 0
            
            for i in range(len(blocks)):
                for j in range(i+1, len(blocks)):
                    # Calculate similarity using normalized cross-correlation
                    similarity = cv2.matchTemplate(blocks[i], blocks[j], cv2.TM_CCOEFF_NORMED)[0][0]
                    if similarity > 0.9:  # High similarity threshold
                        similar_blocks += 1
                    total_comparisons += 1
            
            return similar_blocks / total_comparisons if total_comparisons > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error detecting copy-move forgery: {e}")
            return 0.0
    
    def _detect_splicing(self, image: np.ndarray) -> float:
        """Detect image splicing using edge analysis."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze edge patterns for splicing indicators
            # This is a simplified approach - real implementation would be more sophisticated
            edge_density = np.sum(edges > 0) / edges.size
            
            # High edge density might indicate splicing
            return min(edge_density * 2, 1.0)
            
        except Exception as e:
            logger.error(f"Error detecting splicing: {e}")
            return 0.0
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> float:
        """Analyze compression artifacts."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT to detect JPEG compression artifacts
            # This is a simplified approach
            dct = cv2.dct(np.float32(gray))
            
            # Analyze high-frequency components
            h, w = dct.shape
            high_freq = dct[h//2:, w//2:]
            artifact_score = np.mean(np.abs(high_freq))
            
            return min(artifact_score / 100, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing compression artifacts: {e}")
            return 0.0
    
    def _extract_keyframes(self, video_path: str) -> List[str]:
        """Extract keyframes from video."""
        keyframes = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Extract frames at regular intervals
            interval = max(1, frame_count // 10)  # Extract up to 10 keyframes
            
            for i in range(0, frame_count, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Save keyframe
                    keyframe_path = f"temp_keyframe_{i}.jpg"
                    cv2.imwrite(keyframe_path, frame)
                    keyframes.append(keyframe_path)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting keyframes: {e}")
        
        return keyframes
    
    def _detect_deepfake_indicators(self, video_path: str) -> List[str]:
        """Detect potential deepfake indicators in video."""
        indicators = []
        
        try:
            if self.deepfake_detector is None:
                return indicators
            
            # Extract keyframes and analyze
            keyframes = self._extract_keyframes(video_path)
            
            for keyframe_path in keyframes:
                try:
                    # Load and analyze keyframe
                    image = Image.open(keyframe_path)
                    
                    # Use specialized deepfake detection model
                    result = self.deepfake_detector(image)
                    
                    if result:
                        prediction = result[0]
                        confidence = prediction['score']
                        is_fake = (
                            prediction['label'].lower() in ['fake', 'manipulated', 'deepfake'] or
                            confidence > 0.7  # High confidence threshold
                        )
                        
                        if is_fake:
                            details = f"Confidence: {confidence:.2%}"
                            indicators.append(f'Deepfake indicators detected in frame ({details})')
                    
                except Exception as e:
                    logger.error(f"Error analyzing keyframe: {e}")
            
            # Clean up temporary files
            for keyframe_path in keyframes:
                try:
                    os.remove(keyframe_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error detecting deepfake indicators: {e}")
        
        return indicators
    
    def _analyze_audio(self, video_path: str) -> Dict:
        """Analyze audio track of video."""
        # Placeholder for audio analysis
        # Would implement audio forensics techniques
        return {
            'analysis_available': False,
            'reason': 'Audio analysis not implemented'
        }
    
    def _calculate_image_authenticity_score(self, reverse_results: List[Dict], 
                                          tampering_indicators: List[str], 
                                          metadata: Dict) -> float:
        """Calculate overall authenticity score for image."""
        score = 0.5  # Base score
        
        # Reverse search results
        if reverse_results:
            score += 0.2  # Bonus for finding matches
        
        # Tampering indicators
        if tampering_indicators:
            score -= len(tampering_indicators) * 0.1
        
        # Metadata analysis
        if 'tampering_flags' in metadata and metadata['tampering_flags']:
            score -= len(metadata['tampering_flags']) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_video_authenticity_score(self, reverse_results: List[Dict], 
                                          deepfake_indicators: List[str], 
                                          audio_analysis: Dict) -> float:
        """Calculate overall authenticity score for video."""
        score = 0.5  # Base score
        
        # Reverse search results
        if reverse_results:
            score += 0.2
        
        # Deepfake indicators
        if deepfake_indicators:
            score -= len(deepfake_indicators) * 0.15
        
        return max(0.0, min(1.0, score))
    
    def _calculate_image_confidence(self, reverse_results: List[Dict], 
                                  tampering_indicators: List[str], 
                                  ocr_text: str) -> float:
        """Calculate confidence score for image analysis."""
        confidence = 0.5
        
        if reverse_results:
            confidence += 0.2
        
        if ocr_text:
            confidence += 0.1
        
        if tampering_indicators:
            confidence += 0.1  # More confident when we detect issues
        
        return min(confidence, 1.0)
    
    def _calculate_video_confidence(self, reverse_results: List[Dict], 
                                 deepfake_indicators: List[str], 
                                 keyframe_count: int) -> float:
        """Calculate confidence score for video analysis."""
        confidence = 0.5
        
        if reverse_results:
            confidence += 0.2
        
        if keyframe_count > 5:
            confidence += 0.1
        
        if deepfake_indicators:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_image_verification_status(self, analysis: ImageAnalysis) -> str:
        """Determine verification status for image."""
        if analysis.authenticity_score >= 0.7:
            return "AUTHENTIC"
        elif analysis.authenticity_score <= 0.3:
            return "FAKE"
        elif analysis.tampering_indicators:
            return "SUSPICIOUS"
        else:
            return "UNKNOWN"
    
    def _determine_video_verification_status(self, analysis: VideoAnalysis) -> str:
        """Determine verification status for video."""
        if analysis.authenticity_score >= 0.7:
            return "AUTHENTIC"
        elif analysis.authenticity_score <= 0.3:
            return "FAKE"
        elif analysis.deepfake_indicators:
            return "SUSPICIOUS"
        else:
            return "UNKNOWN"
    
    def _initialize_tampering_detector(self):
        """Initialize tampering detection models."""
        # Placeholder for tampering detection initialization
        return None

# Example usage
async def main():
    verifier = MultiModalVerifier()
    
    # Test image verification
    image_path = r"C:\Users\naroy\Pictures\peakpx.jpg"
    if os.path.exists(image_path):
        verification = await verifier.verify_multimodal_content(image_path, "image")
        
        print(f"Media Type: {verification.media_type}")
        print(f"Verification Status: {verification.verification_status}")
        print(f"Authenticity Score: {verification.overall_authenticity_score:.2f}")
        print(f"Confidence: {verification.confidence:.2f}")
        
        if verification.warnings:
            print(f"Warnings: {', '.join(verification.warnings)}")
        
        if verification.recommendations:
            print(f"Recommendations: {', '.join(verification.recommendations)}")
    
    # Test video verification
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        verification = await verifier.verify_multimodal_content(video_path, "video")
        
        print(f"\nMedia Type: {verification.media_type}")
        print(f"Verification Status: {verification.verification_status}")
        print(f"Authenticity Score: {verification.overall_authenticity_score:.2f}")
        print(f"Confidence: {verification.confidence:.2f}")
        
        if verification.warnings:
            print(f"Warnings: {', '.join(verification.warnings)}")
        
        if verification.recommendations:
            print(f"Recommendations: {', '.join(verification.recommendations)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
