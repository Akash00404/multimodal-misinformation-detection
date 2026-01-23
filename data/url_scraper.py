from newspaper import Article


def extract_text_from_url(url):
    """
    Extract full article text from a news URL
    """
    article = Article(url)
    article.download()
    article.parse()

    text = article.text.strip()

    return text

def is_valid_url_text(text):
    """
    Validate scraped article text
    """
    if not text:
        return False
    if len(text.split()) < 100:
        return False
    return True

if __name__ == "__main__":
    url = "https://www.bbc.com/news/world-europe-60506682"

    text = extract_text_from_url(url)

    if is_valid_url_text(text):
        print("Article extracted successfully\n")
        print(text[:500])
    else:
        print("Failed to extract sufficient article text")
