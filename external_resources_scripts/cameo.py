import pandas as pd
import numpy as np
from ..utils.log_utils import printv


# Setup of GLOBAL VARIABLES related to CAMEO columns. This can change in future versions of the database and a
# correct mapping must be done if so
# These variables are set as global to emphasize the need to check for the existing columns in the database
BASE = ['MVID', 'CAMEO_PRT', 'CAMEO_PRTG', 'CAMEO_INTL']
FACTORS = ['Factor1', 'Factor2', 'Factor3', 'Factor4', 'IncomeFocus']
DECILES = ['Decile1', 'Decile2', 'Decile3', 'Decile4', 'IncomeFocus_Decile']
COORD = ['AvgX', 'AvgY']
EXPENDITURE = {'Total_Expenditure': 'total_expenditure', 'PRT_EXP_1': '1_food', 'PRT_EXP_1_1': '1_1_food_beverages',
               'PRT_EXP_1_1_1':     '1_1_1_cereals', 'PRT_EXP_1_1_2': '1_1_2_meat', 'PRT_EXP_1_1_3': '1_1_3_fish',
               'PRT_EXP_1_1_4':     '1_1_4_milk_cheese_eggs', 'PRT_EXP_1_1_5': '1_1_5_oils',
               'PRT_EXP_1_1_6':     '1_1_6_fruits', 'PRT_EXP_1_1_7': '1_1_7_vegetables', 'PRT_EXP_1_1_8': '1_1_8_sugar',
               'PRT_EXP_1_1_9':     '1_1_9_other_food', 'PRT_EXP_1_2': '1_2_beverages',
               'PRT_EXP_1_2_1':     '1_2_1_coffee_tea', 'PRT_EXP_1_2_2': '1_2_2_water_soft_drinks',
               'PRT_EXP_2':         '2_alcohol_tobacco', 'PRT_EXP_2_1': '2_1_alcohol', 'PRT_EXP_2_2': '2_2_tobacco',
               'PRT_EXP_3':         '3_clothing_footwear', 'PRT_EXP_3_1': '3_1_clothing', 'PRT_EXP_3_2': '3_2_footwear',
               'PRT_EXP_4':         '4_housing', 'PRT_EXP_4_1': '4_1_rents',
               'PRT_EXP_4_1_1':     '4_1_1_rents_main_residence', 'PRT_EXP_4_1_2': '4_1_2_rents_second_residence',
               'PRT_EXP_4_2':       '4_2_subjective_rents', 'PRT_EXP_4_2_1': '4_2_1_subjective_rent_main',
               'PRT_EXP_4_2_2':     '4_2_2_other_subjective', 'PRT_EXP_4_3': '4_3_repair',
               'PRT_EXP_4_3_1':     '4_3_1_repair_material', 'PRT_EXP_4_3_2': '4_3_2_repair_services',
               'PRT_EXP_4_4':       '4_4_other_services_housing', 'PRT_EXP_4_4_1': '4_4_1_water_supply',
               'PRT_EXP_4_4_2':     '4_4_2_solid_waste_collection', 'PRT_EXP_4_4_3': '4_4_3_sanitation_services',
               'PRT_EXP_4_4_4':     '4_4_4_other_housing_services', 'PRT_EXP_4_5': '4_5_electricity_gas_fuels',
               'PRT_EXP_4_5_1':     '4_5_1_electricity', 'PRT_EXP_4_5_2': '4_5_2_gas',
               'PRT_EXP_4_5_4':     '4_5_4_solid_fuels', 'PRT_EXP_5': '5_furniture_household_equipment',
               'PRT_EXP_5_1':       '5_1_furniture', 'PRT_EXP_5_2': '5_2_textile',
               'PRT_EXP_5_3':       '5_3_domestic_base_equipment', 'PRT_EXP_5_4': '5_4_tableware',
               'PRT_EXP_5_5':       '5_5_tools', 'PRT_EXP_5_6': '5_6_maintenance', 'PRT_EXP_6': '6_health',
               'PRT_EXP_6_1':       '6_1_drugs', 'PRT_EXP_6_2': '6_2_health_services', 'PRT_EXP_6_3': '6_3_hospital',
               'PRT_EXP_7':         '7_transportation', 'PRT_EXP_7_1': '7_1_acquisition_vehicles',
               'PRT_EXP_7_2':       '7_2_use_personal_vehicles', 'PRT_EXP_7_3': '7_3_transport_services',
               'PRT_EXP_8':         '8_communications', 'PRT_EXP_8_1': '8_1_postal_services',
               'PRT_EXP_8_2':       '8_2_telecommunication_equipment', 'PRT_EXP_8_3': '8_3_telephone',
               'PRT_EXP_9':         '9_leisure', 'PRT_EXP_9_1': '9_1_audiovisual',
               'PRT_EXP_9_3':       '9_3_other_leisure_equipment', 'PRT_EXP_9_4': '9_4_leisure_cultural_services',
               'PRT_EXP_9_5':       '9_5_books_journals', 'PRT_EXP_9_6': '9_6_travel_tourist',
               'PRT_EXP_10':        '10_education', 'PRT_EXP_10_1': '10_1_education_pre_school',
               'PRT_EXP_10_2':      '10_2_education_basic', 'PRT_EXP_10_4': '10_4_education_higher',
               'PRT_EXP_11':        '11_hotels_restaurants', 'PRT_EXP_11_1': '11_1_catering',
               'PRT_EXP_11_2':      '11_2_lodging', 'PRT_EXP_12': '12_other', 'PRT_EXP_12_1': '12_1_other_health',
               'PRT_EXP_12_3':      '12_3_other_articles', 'PRT_EXP_12_4': '12_4_social_protection',
               'PRT_EXP_12_5':      '12_5_insurance', 'PRT_EXP_12_6': '12_6_financial_services',
               'PRT_EXP_12_7':      '12_7_other_services'}
DICT_HOUSEHOLD = {1: '5. Wealthy Households', 2: '4. Prosperous Households', 3: '3. Comfortable Households',
                  4: '2. Less Affluent Households', 5: '1. Poorer Households', np.nan: np.nan}
DICT_LIFESTAGE = {1: '1. Pre-Family Couples & Singles', 2: '2. Young Couples With Children',
                  3: '3. Families With School Age Children', 4: '4. Older Families & Mature Couples',
                  5: '5. Elders In Retirement', np.nan: np.nan}


def _fill_nulls(cameo, rows_to_fill, verbose=0):
    """Fill the lines in the Cameo data set that are both missing and are necessary, using the values from the closest
    zip code in the same region.

    :param pandas.DataFrame cameo: the original Cameo data set
    :param list|set rows_to_fill: a list of zip codes that should be filled
    :param int verbose: default=0. prints intermediate outputs
    :return: the filled Cameo data set


    FUTURE WORK:
    ------------
    - Vectorize for loops (where possible)
    """

    # Split DataFrame by nulls and filled
    cameo_null = cameo[cameo['CAMEO_PRT'].isna()]
    cameo_fill = cameo[~cameo['CAMEO_PRT'].isna()]

    # Identify null zip codes that are necessary to be filled
    zip_list = list(set(cameo['MVID'][cameo['CAMEO_PRT'].isna()]).intersection(rows_to_fill))

    printv('Filling nulls in the Cameo dataset', level=3, verbose=verbose)
    for i, zip_null in zip(range(len(zip_list)), zip_list):
        # for each zip code without info in the dataset, find the closest zip with info (geographically) and assign
        # this info to the zip code without info
        printv(str(i + 1) + '/' + str(len(zip_list)) + ' - ' + str(zip_null), verbose=verbose, end='\r', level=3)
        distance = []
        zip_to_fill = []

        # future: vectorize
        for zip_fill in [x for x in cameo_fill['MVID'] if x[0:5] == zip_null[0:5]]:
            x = float(cameo_null['AvgX'][cameo_null['MVID'] == zip_null]) - float(
                cameo_fill['AvgX'][cameo_fill['MVID'] == zip_fill])
            y = float(cameo_null['AvgY'][cameo_null['MVID'] == zip_null]) - float(
                cameo_fill['AvgY'][cameo_fill['MVID'] == zip_fill])
            distance.append(x ** 2 + y ** 2)
            zip_to_fill.append(zip_fill)

        # future: vectorize
        for column in [x for x in cameo.columns.tolist() if x not in ['AvgX', 'AvgY']]:
            cameo.loc[cameo['MVID'] == zip_null, column] = cameo_fill.loc[
                cameo_fill['MVID'] == zip_to_fill[distance.index(min(distance))], column].values

    return cameo


def load(cameo_path, cameo_expenditures_path=None, zip_codes=None, fill=False, verbose=0):
    """Function to load cameo database from files

    :param string cameo_path: path to cameo main file
    :param string cameo_expenditures_path: path to cameo_expenditures file
    :param list zip_codes: list of zip_codes to get info for. if "fill" is set to True it will try to fill any null on
    these zip_codes with info from nearby coordinates
    :param bool fill: default: False. Parameter to control if the algorithm should try to fill nulls in database with
    info from nearby coordinates
    :param int verbose: default: 0. Parameter to control verbosity
    :return: DataFrame with Cameo infos
    """

    # Read csv file
    cameo = pd.read_csv(cameo_path, sep=',', encoding='latin-1')

    # Fill nulls based on proximity (if df and column_name are also supplied)
    if fill and (zip_codes is not None):
        try:
            cameo = _fill_nulls(cameo, rows_to_fill=set(zip_codes), verbose=verbose)
            cameo.to_csv(cameo_path, sep=',', encoding='latin-1', index=False)
            printv('CSV updated', level=2, verbose=verbose)
        except:
            printv('Can not fill cameo database due to missing inputs', verbose=verbose)

    # Format nulls as np.nan
    for column in BASE:
        cameo[column] = cameo[column].replace(['X', 'XX'], np.nan)

    for column in FACTORS:
        cameo[column] = cameo[column].replace([-99, -99.99, -99.99999], np.nan)

    for column in DECILES:
        cameo[column] = cameo[column].replace([-99, -99.99, -99.99999], np.nan)

    # Create dictionary structure
    cameo = pd.DataFrame(
        {'cameo_mvid':                cameo.MVID, 'cameo_prt': cameo.CAMEO_PRT,
         'cameo_household':           cameo.CAMEO_INTL.str.slice(0, 1),
         'cameo_lifestage':           cameo.CAMEO_INTL.str.slice(1, 2), 'cameo_families_decile': cameo.Decile1,
         'cameo_families_factor':     cameo.Factor1, 'cameo_education_decile': cameo.Decile2,
         'cameo_education_factor':    cameo.Factor2, 'cameo_industry_decile': cameo.Decile3,
         'cameo_industry_factor':     cameo.Factor3, 'cameo_property_decile': cameo.Decile4,
         'cameo_property_factor':     cameo.Factor4, 'cameo_income_focus': cameo.IncomeFocus,
         'cameo_income_focus_decile': cameo.IncomeFocus_Decile})

    # Map codes
    cameo['cameo_household'] = cameo['cameo_household'].astype(float).map(DICT_HOUSEHOLD)
    cameo['cameo_lifestage'] = cameo['cameo_lifestage'].astype(float).map(DICT_LIFESTAGE)
    cameo['cameo_code'] = cameo['cameo_household'] + ' - ' + cameo['cameo_lifestage']
    cameo = cameo.sort_index(axis=1)

    # Add values from expenditures
    if cameo_expenditures_path is not None:
        cameo_expenditures = pd.read_csv(cameo_expenditures_path, sep='\t', encoding='latin-1')
        cameo_expenditures.rename(index=str, columns=EXPENDITURE, inplace=True)
        cameo_expenditures[list(EXPENDITURE.values())] = cameo_expenditures[list(EXPENDITURE.values())].replace(
            [-99, -99.99, -99.99999], np.nan)
        cameo_expenditures.set_index('PCD7', inplace=True)
        cameo_expenditures = cameo_expenditures[list(EXPENDITURE.values())]

        cameo = pd.merge(cameo, cameo_expenditures, how='left', left_on='cameo_mvid', right_index=True)

    return cameo
