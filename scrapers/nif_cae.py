"""
This file implements company_info() and associated class and functions to assist in data collection using the
portuguese fiscal number (NIF) of a company.

To use this function you can import it using the following command:
>>> from closer_packages.scrapers.nif_cae import get_company_info

Organization: Closer Consulting

Version: 1.0

Date: 2018-09-20
"""

# noinspection PyBroadException

from ..webtools import parse_url_page
import numpy as np


def get_company_info(nif, verbose=0):
    """Obtain information of a company based on its portuguese fiscal number (NIF).

    :param string|int nif: Target NIF for information retrieval.
    :param int verbose: default: 0. Optional. If >0, outputs the intermediate steps into the console with different
        details.
    :return dict: Dictionary with target company info.

    Example:
    >>> company_info = get_company_info(nif = 507246152)
    ... for (key, value) in company_info.items():
    ...     print(key + ': ' + str(value))

    """

    nif_url = 'http://www.nif.pt'
    sicae_url = 'http://www.sicae.pt/Detalhe.aspx?NIPC='

    if nif_url[-1] != '/':
        nif_url = nif_url + '/'

    nif_url = nif_url + str(nif) + '/'
    sicae_url = sicae_url + str(nif)

    nif_soup = parse_url_page(nif_url, verbose=verbose)
    sicae_soup = parse_url_page(sicae_url, verbose=verbose)

    if not len(nif_soup) == 0:
        div = nif_soup.find("div", {"class": "detail"})

        try:
            name = div.find('span', {"class": "search-title"})
            name = name.text
        except Exception:
            name = np.nan

        total_text = []

        for text in div.childGenerator():
            if 'element.NavigableString' in str(type(text)):
                total_text.append(text)

        try:
            zip_code = total_text[3][:8]
        except Exception:
            zip_code = np.nan

        try:
            city = total_text[3][8:].strip()
        except Exception:
            city = np.nan

        try:
            racius_url = nif_soup.find('a', {"class": "link-racius"}).get('href')

            racius_soup = parse_url_page(racius_url, verbose=verbose)

            try:
                c_type = racius_soup.find_all('td', {"class": ''})[0].text
            except Exception:
                c_type = np.nan

            try:
                capital = racius_soup.find_all('td', {"class": ''})[1].text
                if capital == 'Indisponível':
                    capital = np.nan
            except Exception:
                capital = np.nan

            cae = []

            for div in sicae_soup.find_all('div', {'optimal_id': 'letrasCAE'}, title=True):
                cae.append([div.text, div['title']])

            # Work CAE format
            if len(cae) != 0:
                cae_p_code = cae[0][0]
                cae_p_description = cae[0][1]
            else:
                cae_p_code = np.nan
                cae_p_description = np.nan

            try:
                founding_date = racius_soup.find('span', {'itemprop': 'foundingDate'}).text
            except Exception:
                founding_date = np.nan

        except Exception:
            c_type = np.nan
            capital = np.nan
            cae_p_code = np.nan
            cae_p_description = np.nan
            founding_date = np.nan

    else:
        name = np.nan
        zip_code = np.nan
        city = np.nan
        c_type = np.nan
        capital = np.nan
        cae_p_code = np.nan
        cae_p_description = np.nan
        founding_date = np.nan

    result = {'name': name,
              'zip code': zip_code,
              'city': city,
              'type': c_type,
              'capital': capital,
              'cae_code': cae_p_code,
              'cae_description': cae_p_description,
              'founding date': founding_date}

    return result
