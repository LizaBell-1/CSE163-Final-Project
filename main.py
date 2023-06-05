"""
Effects of Population Density on Global Climate Change Indicators
CSE163 AB
Create by Elizaveta Bell, Lauren Yan, Miko Kato
This file contains _
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import geopandas as gpd
from scipy.stats import pearsonr


def reading_csv_files(shp_file: str, population_file: str, co2_file: str,
                      temp_change_file: str, world_pop_file: str) -> \
                      tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame,
                            pd.DataFrame]:
    """
    Reads in and filters datasets to relevant data. Returns a tuple of
    filtered dataframes.
    """
    # Reading in files
    countries_geo = gpd.read_file(shp_file)
    pop_density = pd.read_csv(population_file)
    co2 = pd.read_csv(co2_file)
    temp_change = pd.read_csv(temp_change_file)
    world_pop = pd.read_csv(world_pop_file)

    # Adding 3 character iso code to the geometry dataset
    iso_codes = temp_change.loc[:, ['ISO2', 'ISO3']]
    countries_geo = countries_geo.merge(iso_codes, left_on='ISO',
                                        right_on='ISO2')
    return countries_geo, pop_density, co2, temp_change, world_pop


def pop_density_vs_emissions(year_start: int, year_end: int,
                             pop_df: pd.DataFrame,
                             emissions_df: pd.DataFrame) -> None:
    """
    For a given range of years, creates a scatter plot of the
    carbon dioxide emissions for population density.
    """
    # Selecting year range
    pop_valid_year = (pop_df['Year'] >= year_start) \
        & (pop_df['Year'] <= year_end)
    pop_df_filtered = pop_df[pop_valid_year]
    emissions_valid_years = (emissions_df['year'] >= year_start) \
        & (emissions_df['year'] <= year_end)
    emissions_df_filtered = emissions_df[emissions_valid_years]

    # Merging into one dataframe
    merged = pop_df_filtered.merge(emissions_df_filtered,
                                   left_on='Code', right_on='iso_code')

    # Finding average population density and total carbon emissions
    pop_mean_co2_total = merged.groupby('Code').aggregate(
        {'Population density': 'mean', 'co2': 'sum'}
        )
    pop_mean_co2_total['Population range low'] = \
        (pop_mean_co2_total['Population density']//10)*10

    # Making population ranges
    pop_groups = pop_mean_co2_total.groupby(
        'Population range low').aggregate({'co2': 'sum'})

    # Check validity
    pop_groups['Population range low'] = pop_groups.index[:]
    check_validity(pop_groups, 'Population range low', 'co2',
                   'Pop_Density_vs_Emissions')

    # Making two plots for low and high population density
    sns.relplot(data=pop_groups.loc[:500], x='Population range low', y='co2')
    plt.xlabel('Average population density (people per km^2)')
    plt.ylabel('Total co2 emissions (million tonnes)')
    plt.title('co2 emissions by population density over years '
              + str(year_start) + ' to ' + str(year_end) + ' (low density)')
    plt.savefig('population_density_vs_emissions' + str(year_start) + '_'
                + str(year_end) + '_low.png', bbox_inches='tight')
    sns.relplot(data=pop_groups.loc[510:1700],
                x='Population range low', y='co2')
    plt.xlabel('Average Population Density (people per km^2)')
    plt.ylabel('Total CO2 Emissions (million tonnes)')
    plt.title('CO2 Emissions by Population Density Between '
              + str(year_start) + ' to ' + str(year_end) + ' (high density)')
    plt.savefig('population_density_vs_emissions' + str(year_start) + '_'
                + str(year_end) + '_high.png', bbox_inches='tight')


def pop_density_vs_emissions_country(country: str, code: str,
                                     year_start: int, year_end: int,
                                     pop_df: pd.DataFrame,
                                     emissions_df: pd.DataFrame) -> None:
    """
    For a dictionary of the 10 countries and iso codes with the 5
    highest and 5 lowest population densities and range of years,
    creates a scatter plot of the change in carbon dioxide emissions
    versus population density for each country.
    """
    # Selecting country
    pop_valid_country = (pop_df['Code'] == code)
    pop_df_filtered = pop_df[pop_valid_country]
    emissions_valid_country = (emissions_df['iso_code'] == code)
    emissions_df_filtered = emissions_df[emissions_valid_country]

    # Selecting year
    pop_valid_year = (pop_df['Year'] >= year_start) \
                      & (pop_df['Year'] <= year_end)
    pop_df_filtered = pop_df[pop_valid_year]
    emissions_valid_years = (emissions_df['year'] >= year_start) \
                             & (emissions_df['year'] <= year_end)
    emissions_df_filtered = emissions_df[emissions_valid_years]

    # Merging into one dataframe
    merged = pop_df_filtered.merge(emissions_df_filtered,
                                   left_on='Year', right_on='year')

    # Check validity
    check_validity(merged, 'Population density', 'co2', 'pop_density_vs_emissions_country')

    sns.relplot(data=merged, x='Population density', y='co2')
    plt.xlabel('Population Density (people per km^2)')
    plt.ylabel('CO2 Emissions (million tonnes)')
    plt.title('CO2 Emissions Versus Population Density in ' + country +
              ' Over Years ' + str(year_start) + ' to ' + str(year_end))
    plt.savefig('population_density_vs_emissions_' + country + '_' +
                str(year_start) + '_' + str(year_end) + '.png',
                bbox_inches='tight')


def plot_continent_emissions(year: int, emissions_df: pd.DataFrame,
                             pop_df: pd.DataFrame, world_df: pd.DataFrame,
                             geo_df: gpd.GeoDataFrame) -> None:
    """
    Given a year, geodataframe of world countries, and datasets for
    population density, populations by country, and CO2 emissions,
    plots a heatmap showing emissions and population density
    for each continent.
    """
    # Selecting year
    pop_valid_year = (pop_df['Year'] == year)
    pop_df_filtered = pop_df[pop_valid_year]
    emissions_valid_years = (emissions_df['year'] == year)
    emissions_df_filtered = emissions_df[emissions_valid_years]

    # Filtering and merging dataframes
    emissions_df_filtered = emissions_df_filtered.loc[
        :, ['year', 'iso_code', 'co2']
        ]
    world_population_filtered = world_df.loc[:, ['CCA3', 'Continent']]
    merged_1 = pop_df_filtered.merge(emissions_df_filtered,
                                     left_on='Code', right_on='iso_code')
    merged_2 = merged_1.merge(world_population_filtered,
                              left_on='Code', right_on='CCA3')
    merged = geo_df.merge(merged_2, left_on='ISO3', right_on='Code')
    merged_filtered = merged.loc[:, ['Population density', 'co2',
                                     'Continent', 'geometry']]

    # Dissolving data by continent
    continent_groups = merged_filtered.dissolve(by='Continent',
                                                aggfunc='sum')

    # Plotting heatmap
    geometry = continent_groups['geometry']
    fig, ax = plt.subplots(1)
    geometry.plot(ax=ax, color='#EEEEEE')
    continent_groups.plot(ax=ax, column='co2', legend=True)
    plt.title(str(year) + ' Carbon Dioxide Emissions \n by Continent (Tonnes)')
    plt.savefig('continent_emissions_' + str(year) + '.png',
                bbox_inches='tight')


def find_high_low_pop_density(pop_density: pd.DataFrame) -> pd.DataFrame:
    """
    Given the average population density per entity each year, returns
    a data frame mapping 10 entities with the overall highest and lowest
    population densities to their ISO code.
    """
    countries = {}
    # filter to desired year range
    in_years = pop_density[(pop_density['Year'] >= 1961) &
                           (pop_density['Year'] <= 2022)]
    # calculate mean and sort
    mean_per_country = in_years.groupby(
        ['Entity', 'Code'])['Population density'].mean()
    sort_by_density = mean_per_country.sort_values()
    # select top 5 and bottom 5
    low = sort_by_density.index[:5].tolist()
    high = sort_by_density.index[-5:].tolist()
    # combine into one series
    low_high = low + high
    # transfer to dict format
    for country, code in low_high:
        countries[country] = code

    # turn into series
    countries = pd.Series(countries, name='Code')
    countries.index.name = 'Country'
    countries.reset_index()

    return countries


def filter_temp_and_co2(temp_change: pd.DataFrame, co2: pd.DataFrame) -> \
    "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Pares down temp_change dataset to just columns ISO3, Year,
    and Temp Change and co2_emissions dataset to columns
    iso_code, year, and consumption_co2_per_capita.
    Returns both filtered datasets.
    """
    temps = temp_change.drop(['CTS_Name', 'CTS_Full_Descriptor', 'CTS_Code',
                             'Source', 'Unit', 'Indicator', 'ISO2', 'Country',
                              'ObjectId'], axis=1)
    melt_temps = temps.melt(id_vars='ISO3', var_name='Year',
                            value_name='Temp Change')
    melt_temps['Year'] = melt_temps['Year'].str[-4:].astype(int)

    co2 = co2.loc[:, ['iso_code', 'year',
                                'consumption_co2_per_capita']]
    year = (co2['year'] >= 1961) & (co2['year'] <= 2022)
    co2 = co2[year]

    return melt_temps, co2


def temp_co2_per_country(country: str, temp_change: pd.DataFrame,
                         co2_emissions: pd.DataFrame) -> None:
    """
    Given a country name, plots two scatter plots showing
    the average surface temperature change and CO2 emissions
    from consumption.
    """
    iso = temp_change[temp_change['Country'] == country]
    iso = iso['ISO3'].iloc[0]

    temp_change, co2 = filter_temp_and_co2(temp_change, co2_emissions)
    temp_for_country = temp_change[temp_change['ISO3'] == iso]
    co2_for_country = co2[co2['iso_code'] == iso]

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 10))
    sns.regplot(ax=ax1, data=temp_for_country, x='Year', y='Temp Change')
    ax1.set_title('Average Surface Temperature Change for ' + country, size=18)
    ax1.set_xlabel('Year', size=16)
    ax1.set_ylabel('Average Surface Temperature Change (degrees Celsius)', size=16)

    sns.regplot(ax=ax2, data=co2_for_country, x='year', y='consumption_co2_per_capita')
    ax2.set_title('Consumption-Related CO2 Emissions for ' + country, size=18)
    ax2.set_xlabel('Year', size=16)
    ax2.set_ylabel('Consumption CO2 per Capita (tonnes/person)', size=16)

    plt.savefig('temp_co2_per_country.png')


def temp_vs_co2(temp_change: pd.DataFrame,
                co2_emissions: pd.DataFrame) -> None:
    """
    Given average surface temperature change and co2 emissions by country,
    saves a scatter plot comparing temperature change and consumption-based
    co2 emissions in all countries between 1961-2021.
    """
    # filter temp_change to desired countries using country codes from dict
    # filter using cumulative co2 from consumption per capita
    temp_change, co2 = filter_temp_and_co2(temp_change, co2_emissions)

    # merge datasets, omitting NaN values
    merged = temp_change.merge(co2, left_on=['ISO3', 'Year'],
                               right_on=['iso_code', 'year'], how='inner')

    # check validity
    check_validity(merged, 'consumption_co2_per_capita', 'Temp Change',
                   'Temp_vs_CO2')

    # create plot
    sns.lmplot(data=merged, x='consumption_co2_per_capita', y='Temp Change',
               scatter_kws={"s": 1}, line_kws={"color": "C1"})
    plt.xlabel('CO2 Emissions from Consumption per Capita (Tonnes per Person)',
               size=10)
    plt.ylabel('Average Surface Temperature Change (Degrees Celsius)')
    plt.title('Average Surface Temperature Change vs. CO2 Emissions', size=10)
    plt.savefig('temp_vs_co2.png', bbox_inches='tight')


def check_validity(data: pd.DataFrame, x: str, y: str, title: str) -> float:
    """
    Given the dataset, names of x and y values, and title of dataset,
    prints Pearson correlation coefficient, significance level, and
    p-value between x and y values. Returns p-value.
    """
    data = data.loc[:, [x, y]]
    data = data.dropna()
    sig_level = 0.05 / len(data)
    stat_and_p_value = pearsonr(data[x], data[y])

    print('Validity of ', title, ':')
    print('Correlation Coefficient: ', stat_and_p_value[0])
    print('Significance Level: ', sig_level)
    print('P-value: ', stat_and_p_value[1])

    if stat_and_p_value[1] < sig_level:
        print('Reject the null hypothesis that correlation = 0')
    else:
        print('Fail to reject the null hypothesis that correlation = 0')

    print()

    return float(stat_and_p_value[1])


def combined_dfs(pop_density, temp_change: pd.DataFrame): 
    """
    Merges the population density and temperature change datasets and
    calls on predict_temperature to predict the temperature changes for
    the 10 selected countries.
    """
    temp_change = temp_change.dropna()
    temp_change = temp_change.drop(columns=[
            'ObjectId', 'Country', 'ISO2', 'Indicator',
            'Unit', 'Source', 'CTS_Code', 'CTS_Name',
            'CTS_Full_Descriptor'
    ])

    temp_change = temp_change.rename(columns = {'ISO3':'Code'}, inplace=False)

    result = pd.merge(pop_density, temp_change)

    return result


def predict_temperature(temp_change: pd.DataFrame, country_code: pd.DataFrame):
    """
    Given past temperature changes across various regions, predicts changes
    for a given country in 2023. Takes in a dataframe and the chosen country.
    """

    is_given_country = temp_change['Code'] == country_code
    country_specific_df = temp_change[is_given_country]
    data = country_specific_df.drop(columns=['Code']).T
    data.index = pd.date_range(start='1961', periods=len(data), freq='AS-JAN')

    rmse_list = []
    initial_train_size = 0.5
    step_size = 0.1

    split_points = np.arange(initial_train_size, 1, step_size)
    for split_point in split_points:
        train = data[:int(split_point * (len(data)))]
        valid = data[int(split_point * (len(data))):]

        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        prediction = model_fit.forecast(steps=len(valid))
        rmse = sqrt(mean_squared_error(valid, prediction))
        rmse_list.append(rmse)

    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    prediction_2023 = model_fit.forecast(steps=1)

    return rmse_list, prediction_2023


def forcasted_temp_2023(pop_density: pd.DataFrame, temp_change: pd.DataFrame) -> float:
    """
    Prints the temperature changes and RMS values for each chosen country.
    Takes in two data frames and returns a float.
    """
    given_countries = find_high_low_pop_density(pop_density)
    
    result = combined_dfs(given_countries, temp_change)
    for country_code in result['Code']:
        rmse_list, prediction_2023 = predict_temperature(result, country_code)
        for i, rmse in enumerate(rmse_list):
            print(country_code, ' ', f'RMSE for split point {0.5 + i * 0.1}: {rmse}')

        print('Forecasted temperature change for 2023: ', prediction_2023)

def main():
    countries, pop_density, co2, temp_change, world_pop = \
        reading_csv_files('World_Countries__Generalized_.shp',
                          'population-density (1).csv',
                          'owid-co2-data (3).csv',
                          'Annual_Surface_Temperature_Change (3).csv',
                          'world_population (1).csv')
    
    forcasted_temp_2023(pop_density, temp_change)
    # pop_density_vs_emissions(2012, 2015, pop_density, co2)
    #high_low = find_high_low_pop_density(pop_density)
    # temp_vs_co2(temp_change, co2)
    # predict_temperature(temp_change, countries)
    # plot_continent_emissions(2013, co2, pop_density, world_pop, countries)
    # pop_density_vs_emissions_country('Afghanistan', 'AFG', 2014, 2015, pop_density, co2)
    # temp_co2_per_country('Argentina', temp_change, co2)


if __name__ == '__main__':
    main()