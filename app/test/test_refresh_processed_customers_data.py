# test_customers_json_structure.py
import json
import os
import pytest

import sys
import os
import warnings
warnings.filterwarnings('ignore')
# Get the directory of the current file (test_allfunctions.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (app/)
parent_dir = os.path.dirname(current_dir)

# Get the path to the src directory
src_dir = os.path.join(parent_dir, 'src')

# Add src directory to sys.path
sys.path.append(src_dir)
from utils import *
def test_customers_json_structure():
    # Define the path to the customers.json file
    file_path = '../../raw_data_files/customers.json'

    # Check if the file exists
    assert os.path.exists(file_path), "customers.json file does not exist"

    # Read the JSON file
    with open(file_path, 'r') as file:
        # Load each line as a JSON object
        for line in file:
            customer_data = json.loads(line)

            # Check the top-level structure
            assert 'fraudulent' in customer_data, "Missing 'fraudulent' field"
            assert 'customer' in customer_data, "Missing 'customer' field"
            assert 'orders' in customer_data, "Missing 'orders' field"
            assert 'paymentMethods' in customer_data, "Missing 'paymentMethods' field"
            assert 'transactions' in customer_data, "Missing 'transactions' field"

            # Check customer details
            customer = customer_data['customer']
            assert all(key in customer for key in ['customerEmail', 'customerPhone', 'customerDevice', 'customerIPAddress', 'customerBillingAddress']), "Missing keys in customer details"

            # Check orders structure
            for order in customer_data['orders']:
                assert all(key in order for key in ['orderId', 'orderAmount', 'orderState', 'orderShippingAddress']), "Missing keys in order"

            # Check payment methods structure
            for payment_method in customer_data['paymentMethods']:
                assert all(key in payment_method for key in ['paymentMethodId', 'paymentMethodRegistrationFailure', 'paymentMethodType', 'paymentMethodProvider', 'paymentMethodIssuer']), "Missing keys in payment method"

            # Check transactions structure
            for transaction in customer_data['transactions']:
                assert all(key in transaction for key in ['transactionId', 'orderId', 'paymentMethodId', 'transactionAmount', 'transactionFailed']), "Missing keys in transaction"


import requests_mock
# def get_geolocation(ip_address: str) -> dict:
#     """
#     Fetch geolocation information for a given IP address using the ip-api service.

#     Parameters:
#     ip_address (str): The IP address for which geolocation information is to be fetched.

#     Returns:
#     dict: A dictionary containing the country, countryCode, latitude, and longitude.
#           Returns None for these fields if the API call fails.
#     """
#     try:
#         response = requests.get(f'http://ip-api.com/json/{ip_address}')
#         data = response.json()
#         return {
#             'country': data.get('country', ''),
#             'countryCode': data.get('countryCode', ''),
#             'lat': data.get('lat', ''),
#             'lon': data.get('lon', '')
#         }
#     except Exception as e:
#         # In case of an exception, return None for all fields
#         return {'country': None, 'countryCode': None, 'lat': None, 'lon': None}



def test_get_geolocation_failure():
    # Mock IP for testing
    test_ip = 'invalid_ip'
    
    # Expected data to be returned by the function in case of failure
    expected_data = {
        'country': None,
        'countryCode': None,
        'lat': None,
        'lon': None
    }
    
    with requests_mock.Mocker() as m:
        # Mock the GET request to return a status code indicating failure
        m.get(f'http://ip-api.com/json/{test_ip}', status_code=500)
        
        # Call the function with the mock IP
        result = get_geolocation(test_ip)
        
        # Assert that the returned data matches the expected data
        assert result == expected_data, "The function should return None for all fields in case of failure."

from unittest.mock import patch
import time
import pandas as pd 

# def process_customer_ips(df: pd.DataFrame, customer_column: str) -> pd.DataFrame:
#     """
#     Process a DataFrame to include geolocation information for IP addresses found in a specified column.

#     Parameters:
#     df (pd.DataFrame): The DataFrame containing customer data.
#     customer_column (str): The name of the column in df that contains customer information
#                            in a dictionary format, including the IP address.

#     Returns:
#     pd.DataFrame: The original DataFrame with additional columns for country, countryCode,
#                   latitude, and longitude based on the customer's IP address.
#     """
#     # Extracting customer IP addresses
#     df['customerIPAddress'] = df[customer_column].apply(lambda x: x['customerIPAddress'] if 'customerIPAddress' in x else None)

#     # Rate limit delay
#     rate_limit_delay = 60 / 45  # Adjust based on the API's limit

#     # Fetch geolocation data for each IP address
#     geo_data = []
#     for ip in df['customerIPAddress']:
#         if ip:
#             geo_info = get_geolocation(ip)
#             geo_data.append(geo_info)
#             time.sleep(rate_limit_delay)
#         else: 
#             geo_data.append({'country': None, 'countryCode': None, 'lat': None, 'lon': None})

#     # Create a DataFrame from the geolocation data
#     geo_df = pd.DataFrame(geo_data)

#     # Concatenate the new columns with the original dataframe
#     df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

#     # Optionally, drop the temporary 'customerIPAddress' column if it's no longer needed
#     df.drop('customerIPAddress', axis=1, inplace=True)

#     return df

# Mock data for testing
mock_data = [
    {'customer_info': {'customerIPAddress': '8.8.8.8'}},
    {'customer_info': {'customerIPAddress': '8.8.4.4'}},
    # Add more mock data as needed
]

mock_df = pd.DataFrame(mock_data)

# Expected geolocation information
expected_geo_info = [
    {'country': 'United States', 'countryCode': 'US', 'lat': 37.751, 'lon': -97.822},
    {'country': 'United States', 'countryCode': 'US', 'lat': 37.751, 'lon': -97.822},
    # Add more expected info as needed
]

@patch('utils.get_geolocation')  # Replace with the actual import path
def test_process_customer_ips(mock_get_geo):
    # Setup the mock return values for get_geolocation
    mock_get_geo.side_effect = expected_geo_info

    # Process the DataFrame
    processed_df = process_customer_ips(mock_df, 'customer_info')

    # Verify that the DataFrame has the new columns
    assert 'country' in processed_df.columns
    assert 'countryCode' in processed_df.columns
    assert 'lat' in processed_df.columns
    assert 'lon' in processed_df.columns

    # Verify the values in the DataFrame
    for i, row in processed_df.iterrows():
        assert row['country'] == expected_geo_info[i]['country']
        assert row['countryCode'] == expected_geo_info[i]['countryCode']
        assert row['lat'] == expected_geo_info[i]['lat']
        assert row['lon'] == expected_geo_info[i]['lon']

    # Verify the number of calls to get_geolocation matches the number of IP addresses
    assert mock_get_geo.call_count == len(mock_data)



@pytest.fixture
def mock_requests():
    with requests_mock.Mocker() as m:
        yield m

def test_get_geolocation_from_address_success(mock_requests):
    # Mock address for testing
    test_address = '1600 Amphitheatre Parkway, Mountain View, CA'
    
    # Expected data to be returned by the mocked request
    expected_data = [{
        'display_name': 'Googleplex, Mountain View, Santa Clara County, California, USA',
        'lat': '37.4224764',
        'lon': '-122.0842499'
    }]
    
    # Mock the GET request to return the expected data
    mock_requests.get(f'https://nominatim.openstreetmap.org/search?format=json&q={test_address}', json=expected_data)
    
    # Call the function with the mock address
    result = get_geolocation_from_address(test_address)
    
    # Assert that the returned data matches the expected data
    assert result == {
        'display_name': expected_data[0]['display_name'],
        'lat': expected_data[0]['lat'],
        'lon': expected_data[0]['lon']
    }, "The function should return the expected geolocation data."

def test_get_geolocation_from_address_failure(mock_requests):
    # Mock address for testing
    test_address = 'Nonexistent Place'
    
    # Mock the GET request to simulate no results
    mock_requests.get(f'https://nominatim.openstreetmap.org/search?format=json&q={test_address}', json=[])
    
    # Call the function with the mock address
    result = get_geolocation_from_address(test_address)
    
    # Assert that the function returns None for all fields in case of no results
    assert result == {'display_name': None, 'lat': None, 'lon': None}, "The function should return None for all fields if no results are found."

def test_extract_email_domain():
    assert extract_email_domain('test@example.com') == 'example.com'
    assert extract_email_domain('user@sub.domain.com') == 'sub.domain.com'
    assert extract_email_domain('invalidemail') is None

def test_is_free_domain():
    assert is_free_domain('gmail.com') == 1
    assert is_free_domain('corporate.com') == 0
    assert is_free_domain('yahoo.com') == 1
    assert is_free_domain('') == 0

