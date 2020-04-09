from html.parser import HTMLParser
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

from ..utils.log_utl import printv


class LinkParser(HTMLParser):
    """
    Class that improves HTMLParser to simplify html retrieval.
    # todo add proxy option

    :param dict|None headers: headers to use to replicate an user-agent
    :param int verbose: Optional. default: 0. If >0, outputs the intermediate steps into the console.
    """

    def __init__(self, headers: dict = None, verbose: int = 0) -> None:
        super().__init__()
        self.verbose = verbose
        self.headers = headers or {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                 'Chrome/39.0.2171.95 Safari/537.36'}

    def error(self, message: str) -> None:
        pass

    def get_html(self, url: str) -> str:
        """Method to obtain the html out of a url.

        :param string url: The url to be opened and parsed.
        :return: Parsed raw html.
        """

        printv('Opening URL: ' + url, verbose=self.verbose, level=2)

        # build request
        req = Request(url, headers=self.headers)
        response = urlopen(req)

        if 'text/html' in response.getheader('Content-Type'):
            html_string = response.read().decode("utf-8")
            self.feed(html_string)
            return html_string
        else:
            raise TypeError('URL is not text/html')


def parse_url_page(url: str, verbose: int = 0):
    """Parse url into a BeautifulSoup object.

    :param string url: The url to be opened and parsed.
    :param int verbose: Optional. default: 0. If > 0, outputs the intermediate steps into the console.
    :return: Parsed html as a BeautifulSoup object.
    """

    for _ in range(5):
        try:
            link_parser = LinkParser(verbose=verbose)
            data = link_parser.get_html(url)
        except Exception as e:
            printv(str(e), verbose=verbose)
            if isinstance(e, TypeError):
                break
        else:
            soup = BeautifulSoup(data, 'html.parser')
            printv('Parse successful', verbose=verbose)
            printv(soup.prettify(), level=2, verbose=verbose)
            return soup
    else:
        printv(f'Not able to get_html {url} after 5 attempts', verbose=verbose)

    return
