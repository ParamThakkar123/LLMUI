import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

def crawl_web(query: str) -> str:
    """
    Crawls the web for the given query, fetches the first accessible search result, and returns its text content.
    Skips results where access is denied.
    """
    search_url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    results = soup.find_all("a", class_="result__a")
    if not results:
        return "No results found."

    for result in results:
        first_link = result.get("href")
        parsed = urlparse(first_link)
        qs = parse_qs(parsed.query)
        real_url = qs.get("uddg", [None])[0]
        if real_url:
            real_url = unquote(real_url)
        else:
            if first_link.startswith("//"):
                real_url = "https:" + first_link
            elif first_link.startswith("/"):
                real_url = "https://duckduckgo.com" + first_link
            else:
                real_url = first_link
        try:
            page_resp = requests.get(real_url, headers=headers, timeout=10)
            if page_resp.status_code == 403 or "Access Denied" in page_resp.text or "access denied" in page_resp.text:
                continue
            page_soup = BeautifulSoup(page_resp.text, "html.parser")
            text = page_soup.stripped_strings
            content = " ".join(list(text))
            if content and "Access Denied" not in content and "access denied" not in content:
                return content
        except Exception as e:
            continue

    return "No accessible content found from the search results."

print(crawl_web("What is share price of Nvidia on 7th July 2025 ?"))