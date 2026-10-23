"""
preprocessor.py
Text cleaning and normalization utility for the Fake News Detection System.
Used consistently during both model training and live web inference.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans raw news headline or article text:
    1. Handles non-string/missing inputs.
    2. Removes common wire-service headers (e.g., Reuters prefixes in ISOT dataset).
    3. Lowers case.
    4. Strips URLs, HTML tags, emails.
    5. Strips punctuation and digits while preserving words.
    6. Normalizes whitespaces.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.strip()
    if not text:
        return ""
    
    # 1. Remove Reuters / agency prefixes often found in ISOT real news (e.g., 'WASHINGTON (Reuters) - ')
    # to prevent artificial bias and data leakage
    text = re.sub(r"^.*?\(Reuters\)\s*-\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^.*?\(Reuters\)\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Reuters\)", "", text, flags=re.IGNORECASE)

    # 2. Lowercase
    text = text.lower()

    # 3. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 5. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 6. Remove Twitter/social handles and hashtags symbol
    text = re.sub(r"[@#]\w+", " ", text)

    # 7. Replace punctuation and numbers with space
    text = re.sub(r"[%s\d]" % re.escape(string.punctuation), " ", text)

    # 8. Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    sample = "WASHINGTON (Reuters) - President signs major bill today! Visit https://example.com for #updates."
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))
