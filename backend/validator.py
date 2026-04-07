"""
Grammar-Based Validation for Voice Inventory System

This module handles extraction and validation of structured fields
(product codes, quantities, locations) from transcribed speech.
Uses regex patterns and word-to-number conversion for robust parsing.
"""

import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredFieldExtractor:
    """
    Extracts and validates structured inventory data from speech transcriptions.
    
    Handles various ways workers might speak product codes, quantities, and locations:
    - "Product ABC-123, quantity 50, location A-12"
    - "ABC dash 456, qty fifteen, at B-7"
    - "Code XY-12345, twenty units, zone C-3"
    
    Attributes:
        PRODUCT_CODE_PATTERN: Regex for product codes (e.g., "ABC-123")
        QUANTITY_PATTERN: Regex for quantities (numbers or words)
        LOCATION_PATTERN: Regex for locations (e.g., "A-12")
        CONFIDENCE_THRESHOLD: Minimum confidence for auto-acceptance
    """
    
    # Validation patterns - matches exact format with 2-3 letters + 3-5 digits
    PRODUCT_CODE_PATTERN = r'\b([A-Z]{2,3})-(\d{3,5})\b'
    
    # Quantity patterns - matches numbers or word numbers
    QUANTITY_PATTERN = r'(?:quantity|qty|count|units?|pieces?)?\s*[:\-]?\s*(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)[\s\-]*(one|two|three|four|five|six|seven|eight|nine)?'
    
    # Location patterns - require explicit location markers or strict letter-dash-number format
    # Avoid matching single letters followed by quantities (like "qty 50")
    LOCATION_PATTERN = r'(?:location|loc|zone|at|in|row|aisle)\s*[:\-]?\s*([A-Z])\s*[-–—]?\s*(\d{1,2})\b'
    STRICT_LOCATION_PATTERN = r'\b([A-Z])-(\d{1,2})\b'
    
    # Confidence threshold for auto-acceptance (balanced for accuracy)
    CONFIDENCE_THRESHOLD = 0.60
    
    # Word to number mapping
    WORD_TO_NUMBER = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
        'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
    }
    
    # Phonetic alphabet mapping (for digit-by-digit mode)
    PHONETIC_TO_LETTER = {
        'alpha': 'A', 'bravo': 'B', 'charlie': 'C', 'delta': 'D',
        'echo': 'E', 'foxtrot': 'F', 'golf': 'G', 'hotel': 'H',
        'india': 'I', 'juliet': 'J', 'kilo': 'K', 'lima': 'L',
        'mike': 'M', 'november': 'N', 'oscar': 'O', 'papa': 'P',
        'quebec': 'Q', 'romeo': 'R', 'sierra': 'S', 'tango': 'T',
        'uniform': 'U', 'victor': 'V', 'whiskey': 'W', 'xray': 'X',
        'yankee': 'Y', 'zulu': 'Z'
    }
    
    # Valid location zones (can be customized per warehouse)
    VALID_ZONES = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # Quantity range limits
    MIN_QUANTITY = 1
    MAX_QUANTITY = 9999
    
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        valid_zones: Optional[set] = None
    ):
        """
        Initialize the StructuredFieldExtractor.
        
        Args:
            confidence_threshold: Minimum confidence for auto-acceptance (0-1)
            valid_zones: Set of valid zone letters. Default A-Z.
        """
        self.confidence_threshold = confidence_threshold
        self.valid_zones = valid_zones or self.VALID_ZONES
        
        logger.info(
            f"StructuredFieldExtractor initialized: "
            f"confidence_threshold={confidence_threshold}"
        )
    
    def extract_product_code(self, text: str) -> Optional[str]:
        """
        Extract product code from transcription text.
        
        Matches patterns like:
        - "ABC-123" (standard format)
        - "ABC dash 456" (spoken dash)
        - "XY-12345" (2-3 letters, 3-5 digits)
        - "ABC 123" (space instead of dash)
        
        Args:
            text: Transcription text to search
        
        Returns:
            Formatted product code (e.g., "ABC-123") or None if not found
        
        Examples:
            >>> extractor.extract_product_code("Product ABC-123 here")
            'ABC-123'
            >>> extractor.extract_product_code("Code XY dash 45678")
            'XY-45678'
        """
        if not text:
            return None
        
        # Normalize text: uppercase
        normalized = text.upper()
        
        # Replace spoken "dash" and "hyphen" with actual dash (remove surrounding spaces)
        normalized = re.sub(r'\s*\bDASH\b\s*', '-', normalized)
        normalized = re.sub(r'\s*\bHYPHEN\b\s*', '-', normalized)
        
        # Apply phonetic alphabet conversion for product codes
        # Build potential code from phonetic letters + numbers
        phonetic_code = self._extract_phonetic_product_code(text)
        if phonetic_code:
            return phonetic_code
        
        # Try to match exact product code pattern (with dash)
        match = re.search(self.PRODUCT_CODE_PATTERN, normalized, re.IGNORECASE)
        
        if match:
            letters = match.group(1).upper()
            numbers = match.group(2)
            
            # Validate format
            if 2 <= len(letters) <= 3 and 3 <= len(numbers) <= 5:
                product_code = f"{letters}-{numbers}"
                logger.debug(f"Extracted product code: {product_code}")
                return product_code
        
        # Try alternative patterns (letter sequence + space/dash + number sequence)
        alt_pattern = r'(?:product|code|item|sku)?[\s:]*([A-Z]{2,3})\s+(\d{3,5})\b'
        alt_match = re.search(alt_pattern, normalized)
        
        if alt_match:
            letters = alt_match.group(1).upper()
            numbers = alt_match.group(2)
            if 2 <= len(letters) <= 3 and 3 <= len(numbers) <= 5:
                product_code = f"{letters}-{numbers}"
                logger.debug(f"Extracted product code (alt pattern): {product_code}")
                return product_code
        
        logger.debug(f"No product code found in: {text[:50]}...")
        return None
    
    def _extract_phonetic_product_code(self, text: str) -> Optional[str]:
        """
        Extract product code from phonetic alphabet input.
        
        Handles input like "alpha bravo charlie dash 123" -> "ABC-123"
        """
        normalized = text.lower()
        normalized = re.sub(r'\bdash\b', '-', normalized)
        
        # Find sequence of phonetic words followed by dash and numbers
        words = normalized.split()
        letters = []
        number_part = None
        found_dash = False
        
        for i, word in enumerate(words):
            if word in self.PHONETIC_TO_LETTER:
                if not found_dash:
                    letters.append(self.PHONETIC_TO_LETTER[word])
            elif word == '-' or word == 'dash' or word == 'hyphen':
                found_dash = True
            elif word.isdigit() and found_dash and 3 <= len(word) <= 5:
                number_part = word
                break
            elif word.isdigit() and len(letters) >= 2 and 3 <= len(word) <= 5:
                # Handle case without explicit dash
                number_part = word
                break
        
        if 2 <= len(letters) <= 3 and number_part:
            return f"{''.join(letters)}-{number_part}"
        
        return None
    
    def extract_quantity(self, text: str) -> Optional[int]:
        """
        Extract quantity from transcription text.
        
        Handles:
        - Numeric values: "50", "100"
        - Word numbers: "fifteen", "twenty-three"
        - Compound numbers: "fifty five"
        - With keywords: "quantity 50", "qty 15", "count 100"
        
        Args:
            text: Transcription text to search
        
        Returns:
            Integer quantity or None if not found or out of range (1-9999)
        
        Examples:
            >>> extractor.extract_quantity("quantity 50 units")
            50
            >>> extractor.extract_quantity("twenty-three pieces")
            23
            >>> extractor.extract_quantity("qty fifteen")
            15
        """
        if not text:
            return None
        
        normalized = text.lower()
        
        # First try to find explicit quantity markers
        quantity_patterns = [
            r'(?:quantity|qty|count|units?)[\s:]+(\d+)',
            r'(\d+)\s*(?:units?|pieces?|items?|count)',
            r'(?:quantity|qty|count)[\s:]+([a-z\-\s]+)',
        ]
        
        for pattern in quantity_patterns:
            match = re.search(pattern, normalized)
            if match:
                value = match.group(1).strip()
                
                # Check if it's a number
                if value.isdigit():
                    qty = int(value)
                    if self.MIN_QUANTITY <= qty <= self.MAX_QUANTITY:
                        logger.debug(f"Extracted quantity: {qty}")
                        return qty
                else:
                    # Try to convert word number
                    qty = self._word_to_number(value)
                    if qty and self.MIN_QUANTITY <= qty <= self.MAX_QUANTITY:
                        logger.debug(f"Extracted quantity (word): {qty}")
                        return qty
        
        # Try general word number extraction (handles "twenty-three", "fifty five", etc.)
        word_number_result = self._extract_word_number(normalized)
        if word_number_result and self.MIN_QUANTITY <= word_number_result <= self.MAX_QUANTITY:
            logger.debug(f"Extracted quantity (word number): {word_number_result}")
            return word_number_result
        
        # Try to find standalone numbers that are NOT part of product codes
        # Product codes have format like "ABC-123" or "ABC 123"
        # Remove product code patterns from text before looking for standalone numbers
        text_without_product_codes = re.sub(r'\b[A-Za-z]{2,3}[-\s]?\d{3,5}\b', '', normalized)
        
        # Also remove location patterns (like A-12)
        text_without_locations = re.sub(r'\b[A-Za-z][-\s]?\d{1,2}\b', '', text_without_product_codes)
        
        numbers = re.findall(r'\b(\d{1,4})\b', text_without_locations)
        for num_str in numbers:
            num = int(num_str)
            if self.MIN_QUANTITY <= num <= self.MAX_QUANTITY:
                logger.debug(f"Extracted quantity (fallback): {num}")
                return num
        
        logger.debug(f"No quantity found in: {text[:50]}...")
        return None
    
    def _word_to_number(self, word: str) -> Optional[int]:
        """
        Convert a word number to integer.
        
        Handles:
        - Simple numbers: "fifteen" → 15
        - Compound numbers: "twenty-three" → 23
        - Combined: "fifty five" → 55
        - Hundreds: "one hundred" → 100
        """
        word = word.lower().strip()
        word = word.replace('-', ' ')
        
        # Direct lookup for simple numbers
        if word in self.WORD_TO_NUMBER:
            return self.WORD_TO_NUMBER[word]
        
        # Handle compound numbers
        parts = word.split()
        
        if len(parts) == 2:
            first = self.WORD_TO_NUMBER.get(parts[0])
            second = self.WORD_TO_NUMBER.get(parts[1])
            
            if first is not None and second is not None:
                # e.g., "one hundred" = 1 * 100 = 100
                if second in (100, 1000):
                    return first * second
                # e.g., "twenty three" = 20 + 3
                elif first >= 20 and first < 100 and second < 10:
                    return first + second
        
        elif len(parts) == 3:
            # e.g., "two hundred fifty" or "one hundred twenty three"
            first = self.WORD_TO_NUMBER.get(parts[0])
            second = self.WORD_TO_NUMBER.get(parts[1])
            third = self.WORD_TO_NUMBER.get(parts[2])
            
            if all(x is not None for x in [first, second, third]):
                if second == 100:
                    return first * 100 + third
        
        return None
    
    def _extract_word_number(self, text: str) -> Optional[int]:
        """
        Find and convert word numbers in text.
        """
        # Normalize hyphens in text (e.g., "twenty-three" -> "twenty three")
        normalized_text = text.replace('-', ' ')
        
        # Pattern for compound word numbers like "one hundred", "twenty three"
        compound_pattern = r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(hundred|thousand|one|two|three|four|five|six|seven|eight|nine)\b'
        
        compound_match = re.search(compound_pattern, normalized_text)
        if compound_match:
            word = compound_match.group(0).strip()
            result = self._word_to_number(word)
            if result:
                return result
        
        # Pattern for single word numbers
        word_pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)\b'
        
        match = re.search(word_pattern, normalized_text)
        if match:
            word = match.group(0).strip()
            return self._word_to_number(word)
        
        return None
    
    def extract_location(self, text: str) -> Optional[str]:
        """
        Extract warehouse location from transcription text.
        
        Matches patterns like:
        - "A-5" (standard format)
        - "location B-12"
        - "zone C dash 3"
        - "at D-99"
        - "row E-1"
        
        Args:
            text: Transcription text to search
        
        Returns:
            Formatted location (e.g., "A-12") or None if not found
        
        Examples:
            >>> extractor.extract_location("location A-12")
            'A-12'
            >>> extractor.extract_location("put it in B dash 5")
            'B-5'
        """
        if not text:
            return None
        
        # Normalize text
        normalized = text.upper()
        
        # Replace spoken "dash" and "hyphen" with actual dash (remove surrounding spaces)
        normalized = re.sub(r'\s*\bDASH\b\s*', '-', normalized)
        normalized = re.sub(r'\s*\bHYPHEN\b\s*', '-', normalized)
        
        # Apply phonetic alphabet conversion
        for phonetic, letter in self.PHONETIC_TO_LETTER.items():
            normalized = re.sub(rf'\b{phonetic.upper()}\b', letter, normalized)
        
        # First, try explicit location markers (most reliable)
        match = re.search(self.LOCATION_PATTERN, normalized, re.IGNORECASE)
        
        if match:
            zone = match.group(1).upper()
            number = match.group(2)
            
            # Validate zone
            if zone in self.valid_zones:
                location = f"{zone}-{number}"
                logger.debug(f"Extracted location: {location}")
                return location
        
        # Next, look for strict letter-dash-number format (e.g., "A-12")
        # This avoids matching things like "qty 50" as "Y-50"
        strict_match = re.search(self.STRICT_LOCATION_PATTERN, normalized)
        
        if strict_match:
            zone = strict_match.group(1).upper()
            number = strict_match.group(2)
            
            # Make sure this isn't part of a product code (which has 3-5 digits)
            # Location has 1-2 digits
            if zone in self.valid_zones and len(number) <= 2:
                # Also check context - avoid matching if preceded by quantity keywords
                # Get the position of the match
                start_pos = strict_match.start()
                prefix = normalized[:start_pos].strip()
                
                # Don't match if immediately preceded by quantity keywords
                quantity_keywords = ['QTY', 'QUANTITY', 'COUNT', 'UNITS', 'PIECES']
                if not any(prefix.endswith(kw) for kw in quantity_keywords):
                    location = f"{zone}-{number}"
                    logger.debug(f"Extracted location (strict): {location}")
                    return location
        
        logger.debug(f"No location found in: {text[:50]}...")
        return None
    
    def validate_and_structure(
        self,
        transcription_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract all fields and validate the transcription result.
        
        Performs complete validation:
        1. Check confidence score against threshold
        2. Extract product code, quantity, and location
        3. Validate each field
        4. Return structured result with status
        
        Args:
            transcription_result: Result from SpeechRecognizer containing:
                - 'text': Transcription text
                - 'confidence': Confidence score (0-1)
                - Other fields (optional)
        
        Returns:
            Dict with validation result:
            
            Success case:
            {
                'status': 'success',
                'data': {
                    'product_id': 'ABC-123',
                    'quantity': 50,
                    'location': 'A-12',
                    'confidence': 0.92,
                    'timestamp': datetime
                },
                'raw_text': 'original transcription'
            }
            
            Low confidence case:
            {
                'status': 'low_confidence',
                'retry': True,
                'confidence': 0.65,
                'message': 'Confidence too low, please repeat',
                'raw_text': 'original transcription'
            }
            
            Incomplete case:
            {
                'status': 'incomplete',
                'missing_fields': ['quantity'],
                'found_fields': {
                    'product_id': 'ABC-123',
                    'location': 'A-12'
                },
                'message': 'Missing quantity',
                'raw_text': 'original transcription'
            }
        """
        text = transcription_result.get('text', '')
        confidence = transcription_result.get('confidence', 0.0)
        
        logger.info(f"Validating transcription: '{text[:50]}...' (confidence: {confidence:.2%})")
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.warning(f"Confidence below threshold: {confidence:.2%} < {self.confidence_threshold:.2%}")
            return {
                'status': 'low_confidence',
                'retry': True,
                'confidence': confidence,
                'message': f'Confidence too low ({confidence:.0%}). Please speak more clearly and repeat.',
                'raw_text': text,
                'partial_extraction': {
                    'product_id': self.extract_product_code(text),
                    'quantity': self.extract_quantity(text),
                    'location': self.extract_location(text)
                }
            }
        
        # Extract fields
        product_id = self.extract_product_code(text)
        quantity = self.extract_quantity(text)
        location = self.extract_location(text)
        
        # Check for missing fields
        missing_fields = []
        found_fields = {}
        
        if product_id is None:
            missing_fields.append('product_id')
        else:
            found_fields['product_id'] = product_id
        
        if quantity is None:
            missing_fields.append('quantity')
        else:
            found_fields['quantity'] = quantity
        
        if location is None:
            missing_fields.append('location')
        else:
            found_fields['location'] = location
        
        # Return incomplete status if any field is missing
        if missing_fields:
            missing_str = ', '.join(missing_fields)
            logger.warning(f"Incomplete transcription, missing: {missing_str}")
            return {
                'status': 'incomplete',
                'missing_fields': missing_fields,
                'found_fields': found_fields,
                'message': f'Could not extract: {missing_str}. Please repeat those fields.',
                'raw_text': text,
                'confidence': confidence
            }
        
        # All fields extracted successfully
        logger.info(
            f"Validation successful: product={product_id}, "
            f"qty={quantity}, loc={location}"
        )
        
        return {
            'status': 'success',
            'data': {
                'product_id': product_id,
                'quantity': quantity,
                'location': location,
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat()
            },
            'raw_text': text
        }
    
    def cross_validate_product(
        self,
        product_id: str,
        database: Any
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate that a product ID exists in the database.
        
        Args:
            product_id: Product code to validate
            database: Database instance with product_exists() method
        
        Returns:
            Tuple of (exists: bool, product_details: dict or None)
        
        Example:
            >>> exists, details = extractor.cross_validate_product("ABC-123", db)
            >>> if not exists:
            ...     print("Product not found in database")
        """
        if not product_id:
            return False, None
        
        try:
            exists = database.product_exists(product_id)
            
            if exists:
                details = database.get_product_details(product_id)
                logger.debug(f"Product validated: {product_id}")
                return True, details
            else:
                logger.warning(f"Product not found in database: {product_id}")
                return False, None
                
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return False, None
    
    def suggest_corrections(
        self,
        product_id: str,
        database: Any,
        max_suggestions: int = 3
    ) -> List[str]:
        """
        Suggest similar product codes for potential corrections.
        
        Useful when a product code isn't found - may be a transcription error.
        
        Args:
            product_id: Product code that wasn't found
            database: Database instance with get_all_product_codes() method
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of similar product codes from the database
        """
        if not product_id or not database:
            return []
        
        try:
            all_codes = database.get_all_product_codes()
            
            # Simple similarity: match by prefix or number portion
            letters = product_id.split('-')[0] if '-' in product_id else product_id[:3]
            
            suggestions = []
            for code in all_codes:
                if code.startswith(letters):
                    suggestions.append(code)
                elif code.endswith(product_id.split('-')[-1] if '-' in product_id else ''):
                    suggestions.append(code)
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []
    
    def parse_digit_by_digit(self, text: str) -> str:
        """
        Parse digit-by-digit dictation.
        
        Handles cases where user spells out characters:
        - "A B C dash 1 2 3" → "ABC-123"
        - "alpha bravo charlie dash one two three" → "ABC-123"
        
        Args:
            text: Spoken digit-by-digit input
        
        Returns:
            Assembled string from individual characters
        """
        normalized = text.lower().strip()
        
        # Replace phonetic alphabet
        for phonetic, letter in self.PHONETIC_TO_LETTER.items():
            normalized = normalized.replace(phonetic.lower(), letter)
        
        # Replace word numbers with digits
        digit_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3',
            'four': '4', 'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'nine': '9'
        }
        for word, digit in digit_words.items():
            normalized = normalized.replace(word, digit)
        
        # Replace dash words
        normalized = normalized.replace('dash', '-')
        normalized = normalized.replace('hyphen', '-')
        
        # Remove spaces and other separators (but keep dashes)
        result = ''
        for char in normalized:
            if char.isalnum() or char == '-':
                result += char.upper()
        
        return result


# Create default instance
default_extractor = StructuredFieldExtractor()


def quick_extract(text: str) -> Dict[str, Any]:
    """
    Quick extraction function using default settings.
    
    Args:
        text: Text to extract fields from
    
    Returns:
        Dict with extracted fields
    """
    return {
        'product_id': default_extractor.extract_product_code(text),
        'quantity': default_extractor.extract_quantity(text),
        'location': default_extractor.extract_location(text)
    }


if __name__ == "__main__":
    # Test the extractor
    print("Structured Field Extractor")
    print("=" * 50)
    
    extractor = StructuredFieldExtractor()
    
    # Test cases
    test_cases = [
        "Product ABC-123 quantity 50 location A-12",
        "ABC dash 456, qty fifteen, at B-7",
        "Code XY-12345, twenty units, zone C-3",
        "Put 100 of item DEF-999 in row D-15",
        "Alpha bravo charlie dash one two three, fifty pieces, location echo dash five",
    ]
    
    print("\nTest Cases:")
    print("-" * 50)
    
    for text in test_cases:
        print(f"\nInput: {text}")
        result = quick_extract(text)
        print(f"  Product: {result['product_id']}")
        print(f"  Quantity: {result['quantity']}")
        print(f"  Location: {result['location']}")
    
    # Test validation
    print("\n\nValidation Tests:")
    print("-" * 50)
    
    mock_result = {
        'text': "Product ABC-123 quantity 50 location A-12",
        'confidence': 0.92
    }
    
    validation = extractor.validate_and_structure(mock_result)
    print(f"\nStatus: {validation['status']}")
    if validation['status'] == 'success':
        print(f"Data: {validation['data']}")
