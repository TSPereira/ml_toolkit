import random
from lxml.html import fromstring
import requests

from ..utils.os_utils import check_types


@check_types(https=bool, countries=(list, type(None)), proxy_type=(str, list, type(None)))
def get_proxies(https=True, countries=None, proxy_type='elite proxy'):
    __proxy_types__ = ['elite proxy', 'anonymous', 'transparent']
    if isinstance(proxy_type, str):
        proxy_type = [proxy_type]

    if isinstance(proxy_type, list):
        proxy_type = [p for p in proxy_type if p in __proxy_types__]
        if not proxy_type:
            proxy_type = None

    # Get proxies from this website
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)

    # Create conditions to filter proxies
    cond = '(i.xpath(".//td[7]")[0].text == "yes")' if https else ''
    cond += (' & ' + '(i.xpath(".//td[3]")[0].text in countries)') if countries is not None else ''
    cond += (' & ' + '(i.xpath(".//td[5]")[0].text in proxy_type)') if proxy_type is not None else ''
    cond = cond if len(cond) != 0 else 'True'

    proxies = list()
    for i in parser.xpath('//tbody/tr')[:]:
        try:
            if eval(cond):
                # Grabbing IP and corresponding PORT
                proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
                proxies.append(proxy)
        except IndexError:
            continue

    random.shuffle(proxies)
    return proxies, 'https' if https else 'http'
