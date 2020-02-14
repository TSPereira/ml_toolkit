# -*- coding: utf-8 -*-
from bs4.element import Tag
from ..webtools.url_parsers import parse_url_page
from ..utils import printv, print_progress_bar
from ..sql import PostgresDatabase
import re
import pandas as pd
import numpy as np
import os


# todo finish parsing variables
def _append_dict(old_dict, new_dict):
    return {**old_dict, **new_dict}


def gen_dict_extract(key, var):
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


def switch(val1, val2):
    if val1 > val2:
        return val2, val1
    else:
        return val1, val2


def _expand_on_chr(_df, col_name, char):
    multi = _df[pd.notnull(_df[col_name]) & _df[col_name].str.contains(char)].copy()
    _df = _df.drop(multi.index)
    multi[col_name] = multi[col_name].str.strip().str.split(char)
    multi = pd.DataFrame({_col: np.repeat(multi[_col].values, multi[col_name].str.len()) for _col in
                          multi.columns.difference([col_name])}).assign(
        **{col_name: np.concatenate(multi[col_name].values)})[multi.columns.tolist()]
    _df = pd.concat([_df, multi])
    _df.reset_index(drop=True, inplace=True)

    return _df


def _doors(_df):
    _df = _expand_on_chr(_df, 'Doors', '/')
    _df['Doors'] = _df['Doors'].astype(float)

    return _df


def _seats(_df):
    _df = _expand_on_chr(_df, 'Seats', '/')
    _df.loc[(pd.notnull(_df['Seats'])) & (_df['Seats'].str.contains(r'\+')), 'Seats'] = \
        _df.loc[(pd.notnull(_df['Seats'])) & (_df['Seats'].str.contains(r'\+')), 'Seats'].apply(pd.eval).values
    _df.loc[(pd.notnull(_df['Seats'])) & (_df['Seats'].str.contains('-')), 'Seats'] = _df.loc[
        (pd.notnull(_df['Seats'])) & (_df['Seats'].str.contains('-')), 'Seats'].str.split('-', expand=True)[1]
    _df['Seats'] = _df['Seats'].astype(float)

    return _df


def _gears(_df, col):
    _df[col] = _df[col].replace({'Automatic': ''}, regex=True)
    _cond = _df[col].str.contains(r'(?=\d+/\d+)') & (_df[col].str.len() > 5)

    if not _df[_cond][col].empty:
        words = _df[_cond][col].str.extract(r'/\d+(.*)')
        _df[_cond][col] = _df[_cond][col].str.extract('(.*)/') + ' ' + words + _df[_cond][col].str.extract('(/.*)')

    _df = _expand_on_chr(_df, col, '/')
    _df['Number of Gears'] = _df[col].str.extract(r'(\d+)').astype(float)
    _df['Type of Gearbox'] = _df[col].str.extract(r'(\D+)', expand=False).replace({r'\.': '', '-': ' ', '/': ''},
                                                                                  regex=True).str.strip()
    _df.loc[_df['Type of Gearbox'].isin(['', 'nan']), 'Type of Gearbox'] = np.nan

    # todo join manual and automatic

    return _df


def _get_from_range(col, chars, func):
    _col = col.copy()
    aux = _col.str.strip().replace({'<': '', '>': '', r'\.\.': '.', r'\)': '', r'\.n': 'nan', '‐': '-'}, regex=True).str.split(
        chars, expand=True)

    for c in aux:
        cond = pd.notnull(aux[c]) & aux[c].str.endswith('.')
        aux.loc[cond, c] = aux.loc[cond, c].str[:-1].values.astype(float)

        cond = pd.notnull(aux[c]) & (aux[c] == '')
        aux.loc[cond, c] = 'nan'

    return aux.astype(float).agg(func, axis=1).values


def parse_brand_table(table):
    table.reset_index(drop=True, inplace=True)
    df = table.copy()
    _aux = {'100 km/h - 0': 'm',
            '200 km/h - 0': 'm',
            'Acceleration 0 - 100 km/h': 'sec',
            'Acceleration 0 - 100 km/h (CNG)': 'sec',
            'Acceleration 0 - 100 km/h (LPG)': 'sec',
            'Acceleration 0 - 200 km/h': 'sec',
            'Acceleration 0 - 300 km/h': 'sec',
            'AdBlue tank': 'l',
            'Coolant': 'l',
            'All-electric range': 'km',
            'Approach angle': '°',
            'Climb angle': '°',
            'Departure angle': '°',
            'Ramp angle': '°',
            'Average Energy consumption':  'kWh/100km',
            'Battery capacity': 'kWh',
            'CNG cylinder capacity': 'kg.',
            'CO2 emissions': 'g/km',
            'CO2 emissions (CNG)': 'g/km',
            'CO2 emissions (Ethanol - E85)': 'g/km',
            'CO2 emissions (LPG)': 'g/km',
            'Cylinder Bore': 'mm.',
            'Engine displacement': 'cm3',
            'Engine oil capacity': 'l',
            'Front overhang': 'mm.',
            'Front track': 'mm.',
            'Fuel consumption (economy) - combined': 'l/100km.',
            'Fuel consumption (economy) - combined (CNG)': 'kg/100km',
            'Fuel consumption (economy) - combined (Ethanol - E85)': 'l/100km.',
            'Fuel consumption (economy) - combined (LPG)': 'l/100km.',
            'Fuel consumption (economy) - extra urban': 'l/100km.',
            'Fuel consumption (economy) - extra urban (CNG)': 'kg/100km',
            'Fuel consumption (economy) - extra urban (Ethanol - E85)': 'l/100km.',
            'Fuel consumption (economy) - extra urban (LPG)': 'l/100km.',
            'Fuel consumption (economy) - urban': 'l/100km.',
            'Fuel consumption (economy) - urban (CNG)': 'kg/100km',
            'Fuel consumption (economy) - urban (Ethanol - E85)': 'l/100km.',
            'Fuel consumption (economy) - urban (LPG)': 'l/100km.',
            'Fuel tank volume': 'l',
            'Fuel tank volume (LPG)': 'l',
            'Height': 'mm.',
            'Kerb Weight': 'kg.',
            'Length': 'mm.',
            'Max. roof load': 'kg.',
            'Max. weight': 'kg.',
            'Maximum engine speed': 'rpm.',
            'Maximum speed': 'km/h',
            'Maximum speed (CNG)': 'km/h',
            'Maximum speed (LPG)': 'km/h',
            'Maximum volume of Luggage (trunk)': 'l',
            'Minimum turning circle (turning diameter)': 'm',
            'Minimum volume of Luggage (trunk)': 'l',
            'Permitted towbar download': 'kg.',
            'Permitted trailer load with brakes (12%)': 'kg.',
            'Permitted trailer load with brakes (8%)': 'kg.',
            'Permitted trailer load without brakes': 'kg.',
            'Piston Stroke': 'mm.',
            'Rear (Back) track': 'mm.',
            'Rear overhang': 'mm.',
            'Ride height': 'mm.',
            'Wading depth': 'mm.',
            'Wheelbase': 'mm.',
            'Width': 'mm.',
            'Width including mirrors': 'mm.',
            'Width with mirrors folded': 'mm.',
            'Year of putting into production': 'year',
            'Year of stopping production': 'year',
            'Compression ratio': '',
            'Number of cylinders': '',
            'Number of valves per cylinder': '',
            }

    _new_names = {}
    for _col in _aux:
        if _col in df.columns:
            _new_names[_col] = _col + ' (' + _aux[_col].strip() + ')'
            df[_col] = df[_col].astype(str).replace({'\xa0': '', '\u202f': '', '\x80': '', '\x93': ''},
                                                    regex=True).str.replace(' ', '').str.replace(_aux[_col], '').str.strip()
            try:
                df[_col] = pd.to_numeric(df[_col], errors='coerce')

            except Exception:
                pass

    if 'ABS' in df.columns:
        df['ABS'] = df['ABS'].fillna(False).replace('yes ', True, regex=True).astype(bool)

    if 'Coupe type' in df.columns:
        df[['BodyType', 'BodySubType']] = df['Coupe type'].str.split(',', expand=True)
        df.drop(['Coupe type'], axis=1, inplace=True)

    if 'Drive wheel' in df.columns:
        df['Drive wheel'] = df['Drive wheel'].replace({'Rear wheel drive ': 'RWD', 'Front wheel drive ': 'FWD',
                                                       'All wheel drive (4x4) ': 'AWD'})

    if 'Fuel Type' in df.columns:
        df['Fuel Type'] = df['Fuel Type'].str.replace('(Gasoline)', '').replace({'Petrol': 'Gasoline', 'petrol': 'Gasoline'},
                                                                                regex=True)

    if 'Position of engine' in df.columns:
        df[['Position of engine', 'Position of engine (direction)']] = df['Position of engine'].str.split(',', expand=True)

    # Fuel correction
    if 'Fuel consumption (economy) - combined' in df.columns:
        df['Fuel consumption (economy) - combined'] = df['Fuel consumption (economy) - combined'].replace(
            '4.14.3', '4.1-4.3', regex=True)

    if 'Maximum speed' in df.columns:
        df['Maximum speed'] = df['Maximum speed'].str.replace('+', '')

    if 'Minimum volume of Luggage (trunk)' in df.columns:
        _cond = (pd.notnull(df['Minimum volume of Luggage (trunk)'])) & \
                (df['Minimum volume of Luggage (trunk)'].str.contains(r'\+'))
        df.loc[_cond, 'Minimum volume of Luggage (trunk)'] = df.loc[_cond, 'Minimum volume of Luggage (trunk)'].apply(
            pd.eval).values

    if 'Minimum turning circle (turning diameter)' in df.columns:
        df['Minimum turning circle (turning diameter)'] = df['Minimum turning circle (turning diameter)'].replace('4WS', '',
                                                                                                                  regex=True)

    # Fuel tank volume
    if 'Fuel tank volume' in df.columns:
        df['Fuel tank volume'] = df['Fuel tank volume'].replace({'optiona': '', '4WS': '', 'л': '', 'x': '*'}, regex=True)

        _cond = (pd.notnull(df['Fuel tank volume'])) & (df['Fuel tank volume'].str.contains(r'^(?=.*\()(?=.*\))(?=.*LPG)'))
        df.loc[_cond, ['Fuel tank volume', 'Fuel tank volume (LPG)']] = \
            df.loc[_cond, 'Fuel tank volume'].str.extract(r'(\d+)\(.*?(\d+).*LPG\)').values

        _cond = (pd.notnull(df['Fuel tank volume'])) & (df['Fuel tank volume'].str.contains(r'\+|\*'))
        df.loc[_cond, 'Fuel tank volume'] = df.loc[_cond, 'Fuel tank volume'].apply(pd.eval).values

    # Deal with ranges
    # Angles
    ang = ['Approach angle', 'Climb angle', 'Departure angle', 'Ramp angle']
    for _col in ang:
        if _col in df.columns:
            df['Min ' + _col] = _get_from_range(df[_col].str.replace('off-road', ''), r'/|-|\(', 'min')
            df['Max ' + _col] = _get_from_range(df[_col].str.replace('off-road', ''), r'/|-|\(', 'max')
            df.drop(_col, axis=1, inplace=True)

    _aux = {'Front track': (r'/|-|\(', 'max'),
            'Height': (r'/|-|\(', 'min'),
            'Kerb weight': (r'/|-|\(', 'min'),
            'Max. weight': (r'/|-|\(', 'max'),
            'Max. roof load': (r'/|-|\(', 'max'),
            'Length': (r'/|-|\(', 'min'),
            'Rear (Back) track': (r'/|-|\(', 'max'),
            'Ride height': (r'/|-|\(', 'min'),
            'Width': (r'/|-|\(', 'min'),
            'Width including mirrors': (r'/|-|\(', 'max'),
            'Wading depth': (r'/|-|\(', 'min'),
            'Wheelbase': (r'/|-|\(', 'min'),
            'Rear overhang': (r'/|-|\(', 'min'),
            'Permitted towbar download': (r'/|-|\(', 'min'),
            'Permitted trailer load with brakes (12%)': (r'/|-|\(', 'min'),
            'Permitted trailer load with brakes (8%)': (r'/|-|\(', 'min'),
            'Permitted trailer load without brakes': (r'/|-|\(', 'min'),
            'Acceleration 0 - 100 km/h': ('-', 'mean'),
            'Acceleration 0 - 100 km/h (CNG)': ('-', 'mean'),
            'Acceleration 0 - 100 km/h (LPG)': ('-', 'mean'),
            'Acceleration 0 - 200 km/h': ('-', 'mean'),
            'Acceleration 0 - 300 km/h': ('-', 'mean'),
            'All-electric range': (r'/|-|\(', 'min'),
            'Average Energy consumption': (r'/|-|\(', 'max'),
            'CO2 emissions': (r'/|-|\(', 'max'),
            'CO2 emissions (CNG)': (r'/|-|\(', 'max'),
            'CO2 emissions (Ethanol - E85)': (r'/|-|\(', 'max'),
            'CO2 emissions (LPG)': (r'/|-|\(', 'max'),
            'Fuel consumption (economy) - combined': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - combined (CNG)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - combined (Ethanol - E85)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - combined (LPG)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - extra urban': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - extra urban (CNG)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - extra urban (Ethanol - E85)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - extra urban (LPG)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - urban': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - urban (CNG)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - urban (Ethanol - E85)': (r'/|-|\(', 'mean'),
            'Fuel consumption (economy) - urban (LPG)': (r'/|-|\(', 'mean'),
            'Drag coefficient': ('-|/', 'mean'),
            'Maximum engine speed': (r'/|-|\(', 'max'),
            'Maximum speed': (r'/|-|\(', 'max'),
            'Maximum volume of Luggage (trunk)': (r'/|-|\(', 'mean'),
            'Minimum volume of Luggage (trunk)': (r'/|-|\(', 'min'),
            'Minimum turning circle (turning diameter)': (r'/|-|\(', 'min'),
            'Fuel tank volume': (r'/|-|\(', 'min')}

    for _col in _aux:
        if _col in df.columns:
            df[_col] = _get_from_range(df[_col], _aux[_col][0], _aux[_col][1])

    # Parse columns that produce augmentation in the dataset
    if 'Doors' in df.columns:
        df = _doors(df)

    if 'Seats' in df.columns:
        df = _seats(df)

    if 'Number of Gears (manual transmission)' in df.columns:
        df = _gears(df, 'Number of Gears (manual transmission)')

    if 'Number of Gears (automatic transmission)' in df.columns:
        df = _gears(df, 'Number of Gears (automatic transmission)')

    if ['Number of Gears (manual transmission)', 'Number of Gears (automatic transmission)'] in df.columns:
        # join both Number of gears
        pass

    # String columns
    _aux = {'Brand': str,
            'BodyType': str,
            'BodySubType': str,
            'DriveWheel': str,
            'Front brakes': str,
            'Front suspension': str,
            'Fuel System': str,
            'FuelType': str,
            'Generation': str,
            'Model': str,
            'Model Engine': str,
            'Number of Gears (automatic transmission)': str,
            'Number of Gears (manual transmission)': str,
            'Position of cylinders': str,
            'Pos. of engine': str,
            'Position of engine (direction)': str,
            'Power steering': str,
            'Rear brakes': str,
            'Rear suspension': str,
            'Steering type': str,
            'Tire size': str,
            'Turbine': str,
            'Wheel rims size': str}

    for _col in _aux:
        if _col in df.columns:
            df[_col] = df[_col].astype(_aux[_col]).str.strip()

    # df.rename(columns=_new_names, inplace=True)
    print(1)

    return df


def structure_brand(_brand, verbose=0):
    brand_table = pd.DataFrame()
    all_data = gen_dict_extract('data', _brand)
    printv('', verbose=verbose)

    for data in all_data:
        printv(data.values[3], verbose=verbose)
        _data = data.to_frame().transpose()
        _data.columns = _data.columns.str.strip()

        brand_table = pd.concat([brand_table, _data])

    return brand_table


def _rename_table(name):
    return name.translate({'-': '_', ' ': '_', '.': ''}).lower()


def _check_str_or_list(var):
    if type(var) == list:
        return var

    elif type(var) == str:
        return [var]

    else:
        raise TypeError('variable passed is not str or list.')


def extract_database(_save_mode='sql', _save_path=None, _brands_to_include=None, _brands_to_exclude=None, verbose=0):
    assert _save_mode in ('sql', 'csv'), '"save_mode" must be on of "sql" or "csv".'
    if (_save_mode == 'csv') & (_save_path is None):
        raise KeyError('When "save_mode=\'csv\'", "save_path" must be set.')

    website = 'https://www.auto-data.net'
    all_brands_ext = '/en/allbrands'
    auto = False

    if _save_mode == 'sql':
        auto = PostgresDatabase('Automotive')

    _brands_to_exclude = [] if _brands_to_exclude is None else _check_str_or_list(_brands_to_exclude)
    _brands_to_include = [] if _brands_to_include is None else _check_str_or_list(_brands_to_include)

    printv('Fetching brand pages...', verbose=verbose)
    main_page = parse_url_page(website + all_brands_ext)
    brands = {_brand.text: {'url': _brand.get('href')} for _brand in main_page.find_all('a', {'class': 'marki_blok'})}
    not_retrieved = {'brand': [], 'gama': [], 'model': []}

    if _brands_to_include:
        _brands = set(_brands_to_include).intersection(brands).difference(_brands_to_exclude)
    else:
        _brands = set(brands).difference(_brands_to_exclude)

    _i = 0
    for _brand in _brands:
        _i += 1

        printv('\n'+_brand, verbose=verbose)
        brands[_brand]['type'] = 'brand'

        brand_url = website + brands[_brand]['url']
        brand_page = parse_url_page(brand_url)

        if not brand_page:
            not_retrieved['brand'].append((_brand, brand_url))
            continue

        brands[_brand] = _append_dict(brands[_brand], {gama.text.strip(): {'url': gama.get('href')}
                                                       for gama in brand_page.find_all('a', {'class': 'modeli'})})

        for gama in brands[_brand]:
            if gama not in ('url', 'type'):
                printv(gama, level=2, verbose=verbose)
                brands[_brand][gama]['type'] = 'gama'

                gama_url = website + brands[_brand][gama]['url']
                gama_page = parse_url_page(gama_url)

                if not gama_page:
                    not_retrieved['gama'].append((gama, gama_url))
                    continue

                generations = [gen.get_text() for gen in gama_page.find_all('h2')]
                gen_tables = gama_page.find_all('table', {'class': 'carData colorbl'})

                for generation, table in zip(generations, gen_tables):
                    printv(generation, level=3, verbose=verbose)
                    brands[_brand][gama][generation] = {'type': 'generation'}
                    _aux = {url.get_text(): _append_dict({'type': 'model'}, {'url': url.get('href')})
                            for url in table.find_all('a')}
                    brands[_brand][gama][generation] = _append_dict(brands[_brand][gama][generation], _aux)

                    for model in brands[_brand][gama][generation]:
                        if model not in ('url', 'type'):
                            printv(model, level=4, verbose=verbose)

                            model_url = website + brands[_brand][gama][generation][model]['url']
                            model_page = parse_url_page(model_url)

                            if not model_page:
                                not_retrieved['model'].append((model, model_url))
                                continue

                            regex = re.compile('b*')
                            table = model_page.find_all('tr', {'class': regex})
                            _tags = dict(((col.get_text() for col in row.contents if type(col) == Tag) for row in table))
                            brands[_brand][gama][generation][model]['data'] = pd.Series(_tags)

        structured_brand = structure_brand(brands[_brand], verbose=verbose)

        if (_save_mode == 'sql') & auto:
            auto.upload_table(_rename_table(_brand), structured_brand)
        elif _save_mode == 'csv':
            structured_brand.to_csv(_save_path + '/' + _brand + '.csv')
            printv('Saved {}\n'.format(_brand), verbose=verbose)

        print_progress_bar(_i, len(_brands), verbose=verbose)
    return brands


if __name__ == '__main__':
    get_from_website = False

    if get_from_website:
        save_mode = 'sql'
        save_path = 'data/vehicle_database'
        brands_to_include = None  # Actually means all
        brands_to_exclude = None

        database = extract_database(_save_mode=save_mode, _save_path=save_path, _brands_to_include=brands_to_include,
                                    _brands_to_exclude=brands_to_exclude, verbose=10)

    else:
        brand_files = os.listdir('data/vehicle_database/')
        common_df = pd.DataFrame()
        for i in range(len(brand_files)):
            print(brand_files[i])
            brand = pd.read_csv('data/vehicle_database/'+brand_files[i])
            # parsed_brand = parse_brand_table(brand)
            common_df = pd.concat([common_df, brand])

        parse_brand_table(common_df)
        print(1)
