"""
Title
CSE163 A
Create by Elizaveta Bell, Lauren Yan, Miko Kato
This file contains the test cases for main.py.
"""

import main as m
import pandas as pd
import math


def test_find_high_low_pop_density() -> None:
    """
    Tests find_high_low_pop_density function
    """
    pop_density_test = pd.read_csv('/home/test_docs/pop_density_test.csv')

    assert m.find_high_low_pop_density(pop_density_test) == {
        'South Sudan': 'SSD', 'Iran': 'IRN', 'Spain': 'ESP',
        'Albania': 'ALB', 'Indonesia': 'IDN', 'Sri Lanka': 'LKA',
        'South Korea': 'KOR', 'India': 'IND', 'Netherlands': 'NLD',
        'Hong Kong': 'HKG'
        }


def test_check_validity() -> None:
    """
    Tests check_validity function
    """
    co2 = pd.read_csv('/home/test_docs/co2.csv')

    a = m.check_validity(co2, 'population', 'gdp', 'Test1')
    b = m.check_validity(co2, 'year', 'year2', 'Test2')

    # Fail to reject
    assert math.isclose(a, 0.337669, abs_tol=abs(a - 0.337669))
    # Reject
    assert math.isclose(b, 1.0, abs_tol=abs(b - 1.0))


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
    test_find_high_low_pop_density()
    test_check_validity()


if __name__ == '__main__':
    main()