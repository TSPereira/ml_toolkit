from html.parser import HTMLParser
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from ..utils.log_utils import printv


class LinkParser(HTMLParser):
    """Class that improves HTMLParser to simplify html retrieval."""

    def __init__(self, headers=None):
        super().__init__()
        if headers is None:
            self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                                          'Chrome/39.0.2171.95 Safari/537.36'}

    def error(self, message):
        pass

    def get_html(self, url, verbose=0):
        """Method to obtain the html out of a url.

        :param string url: The url to be opened and parsed.
        :param int verbose: Optional. default: 0. If >0, outputs the intermediate steps into the console.
        :return: Parsed raw html.
        """

        printv('Opening URL: ' + url, verbose=verbose)

        # build request
        req = Request(url, headers=self.headers)

        response = urlopen(req)
        if 'text/html' in response.getheader('Content-Type'):
            html_bytes = response.read()
            html_string = html_bytes.decode("utf-8")
            self.feed(html_string)
            return html_string
        else:
            printv('URL is not text/html', verbose=verbose)
            return "", []


def parse_url_page(url, verbose=0):
    """Parse url into a BeautifulSoup object.

    :param string url: The url to be opened and parsed.
    :param int verbose: Optional. default: 0. If >0, outputs the intermediate steps into the console.
    :return: Parsed html as a BeautifulSoup object.
    """

    for _ in range(5):
        try:
            link_parser = LinkParser()
            data = link_parser.get_html(url, verbose=verbose)
        except Exception:
            pass
        else:
            soup = BeautifulSoup(data, 'html.parser')
            printv('Parse successful', verbose=verbose)
            printv(soup.prettify(), level=2, verbose=verbose)
            return soup
    else:
        printv('Not able to get_html ' + url, verbose=verbose)
        return []
