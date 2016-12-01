from BeautifulSoup import BeautifulSoup
import urllib2


def get_trending_links(language=None, since="daily"):
    """Return a list of trending GitHub repositories for the specified parameters.

    Args:
        language (None, optional): The programming language. If not set, all languages are included
        since (str, optional): One of ['daily', 'weekly', 'monthly']

    Returns:
        TYPE: List of trending repositories (e.g. ['/rails/rails'])
    """
    language_param = "/" + language if language else ""
    url = "https://github.com/trending{language}?since={since}".format(language=language_param, since=since)
    html_page = urllib2.urlopen(url)
    links = []
    soup = BeautifulSoup(html_page)
    for link_tag in soup.findAll('a'):
        link = link_tag["href"]
        if link is not None and link.startswith("/") and len(link.split("/")) == 3:
            links.append(link)
    return links
