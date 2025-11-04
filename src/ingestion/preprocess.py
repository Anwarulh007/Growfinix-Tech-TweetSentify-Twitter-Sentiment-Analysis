import re
import html

def clean_tweet(text: str) -> str:
    """
    Clean tweet text by removing URLs, mentions, hashtags, HTML entities, and extra spaces.
    """
    text = html.unescape(text)
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)          # Remove mentions
    text = re.sub(r"#", "", text)             # Remove hashtag symbol
    text = re.sub(r"\s+", " ", text)          # Remove extra whitespace
    return text.strip()
