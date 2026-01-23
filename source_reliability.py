from urllib.parse import urlparse


# Predefined source reliability scores
SOURCE_RELIABILITY_MAP = {
    "bbc.com": 0.95,
    "reuters.com": 0.97,
    "theguardian.com": 0.92,
    "nytimes.com": 0.94,
    "washingtonpost.com": 0.93,
    "cnn.com": 0.90,
    "ndtv.com": 0.90,
    "indiatoday.in": 0.88,
    "thehindu.com": 0.92,
    "timesofindia.indiatimes.com": 0.85
}


def get_source_reliability(url):
    """
    Returns reliability score based on news source domain
    """
    try:
        domain = urlparse(url).netloc.lower()

        # Remove 'www.' if present
        if domain.startswith("www."):
            domain = domain.replace("www.", "")

        return SOURCE_RELIABILITY_MAP.get(domain, 0.6)  # default for unknown sources

    except Exception:
        return 0.6
