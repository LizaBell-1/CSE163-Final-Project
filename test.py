"""
Title
CSE163 AB
Create by Elizaveta Bell, Lauren Yan, Miko Kato
This file contains the test cases for main.py.
"""

import main as m
import pandas as pd
import math


def test_find_high_low_pop_density(pop_density_test: pd.DataFrame) -> None:
    """
    Tests find_high_low_pop_density function
    """

    assert m.find_high_low_pop_density(pop_density_test) == {
        'South Sudan': 'SSD', 'Iran': 'IRN', 'Spain': 'ESP',
        'Albania': 'ALB', 'Indonesia': 'IDN', 'Sri Lanka': 'LKA',
        'South Korea': 'KOR', 'India': 'IND', 'Netherlands': 'NLD',
        'Hong Kong': 'HKG'
        }


def test_check_validity(co2_test: pd.DataFrame) -> None:
    """
    Tests check_validity function
    """

    a = m.check_validity(co2_test, 'population', 'gdp', 'Test1')
    b = m.check_validity(co2_test, 'year', 'year', 'Test2')

    # Fail to reject
    assert math.isclose(a, 0.337669, abs_tol=abs(a - 0.337669))
    # Reject
    assert math.isclose(b, 1.0, abs_tol=abs(b - 1.0))


def test_temp_vs_co2(co2_test: pd.DataFrame, temp_test: pd.DataFrame) -> None:
    """
    Tests temp_vs_co2 function
    """
    #m.temp_vs_co2(co2_test, temp_test)


def test_temp_co2_per_country() -> None:
    """
    Tests temp_co2_per_country function
    """


def test_predict_temp():
    """
    Tests predict_temperature function
    """

def test_pop_density_vs_emissions():
    """
    Tests the pop_density_vs_emissions function.
    """
    

def test_pop_density_vs_emissions_country():
    """
    Tests the pop_density_vs_emissions_country function.
    """


def test_plot_continent_emissions():
    """
    Tests the plot_continent_emissions function.
    """


def main():
    pop_density_test = pd.read_csv(r'C:\Users\mikok\CSE163-Final-Project\test_docs\pop_density_test.csv')
    co2_test = pd.read_csv(r'C:\Users\mikok\CSE163-Final-Project\test_docs\co2_test.csv')
    temp_test = pd.read_csv(r'C:\Users\mikok\CSE163-Final-Project\test_docs\temp_test.csv')

    test_find_high_low_pop_density(pop_density_test)
    test_check_validity(co2_test)
    test_temp_vs_co2(co2_test, temp_test)


if __name__ == '__main__':
    main()