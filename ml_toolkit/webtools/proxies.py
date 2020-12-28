import requests
from random import choice
from contextlib import suppress

from proxyscrape import create_collector


class ProxyManager:
    def __init__(self, protocol='https', anonymity=True, autorefresh=True, max_refresh_tries=5,
                 check_url='https://www.google.com', check_timeout=2, limit=-1, verbose=0):
        self.protocol = protocol
        self.anonymity = anonymity
        self._collector = create_collector('default', protocol)
        self._collector.apply_filter(dict(anonymous=anonymity))
        self.autorefresh = autorefresh
        self._max_refresh_tries = max_refresh_tries
        self._auto_refresh_counter = 0
        self.check_timeout = check_timeout
        self._check_url = check_url
        self.limit = limit
        self.verbose = verbose

        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                                      'Chrome/87.0.4280.66 Safari/537.36'}
        self.proxies = []
        self._proxies = []

        self.refresh_proxies()

    def _get_proxy(self, random=False, check_proxy=True):
        # check active list of proxies and if needed/possible update it
        while not self._proxies:
            if self._auto_refresh_counter < self._max_refresh_tries:
                if self.autorefresh:
                    self.refresh_proxies()
                else:
                    self._proxies = self.proxies.copy()

                self._auto_refresh_counter += 1
            else:
                raise requests.exceptions.ProxyError('Could not find a working proxy.')

        # get the index to take from the list, take the proxy and update the active list
        i = 0 if not random else choice(range(len(self._proxies)))
        proxy = self._proxies.pop(i)

        # check the proxy
        if check_proxy:
            if not self._check_proxy(proxy):
                proxy = self._get_proxy(random, check_proxy)

        return proxy

    def get_proxy(self, random=False, check_proxy=True):
        if self.verbose:
            print('Finding a proxy...')

        self._auto_refresh_counter = 0
        return self._get_proxy(random, check_proxy)

    def get_proxy_dict(self, random=False, check_proxy=True):
        return {self.protocol: self.get_proxy(random, check_proxy)}

    def refresh_proxies(self):
        self.proxies = self._get_proxies()
        self._proxies = self.proxies.copy()

    def _get_proxies(self):
        self._collector.refresh_proxies(force=True)
        proxies = list(map(lambda x: f'{x.host}:{x.port}', self._collector.get_proxies()))
        limit = len(proxies) if self.limit < 0 else self.limit
        return proxies[:limit]

    def _check_proxy(self, proxy):
        proxy = dict(https=f'{self.protocol}://{proxy}')

        with suppress(Exception):
            resp = requests.get(self._check_url, headers=self.headers, proxies=proxy, timeout=self.check_timeout)
            return resp.status_code == 200

        return False


# def _get_proxies(self, protocol='https', anonymity='High'):
#     # todo add to proxyscrape as additional resource
#     # add all pages?
#     # limit requests to self.limit
#     url = 'https://hidemy.name/en/proxy-list/?type=s'
#     resp = requests.get(url, headers=self.headers)
#
#     soup = BeautifulSoup(resp.text, 'lxml')
#     table = soup.find('table')
#
#     df = pd.read_html(str(table))[0]
#     df = df[(df['Type'].str.contains(protocol.upper())) & (df['Anonymity'] == anonymity)]
#     proxies = df['IP address'].str.cat(df['Port'].astype(str), sep=':').to_list()
#     return proxies
