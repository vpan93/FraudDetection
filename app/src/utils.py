import json
import pandas as pd
import requests
import time
import urllib.parse
from collections import Counter
import numpy as np
# def read_json_to_dataframe(file_path: str) -> pd.DataFrame:
#     """
#     Reads a JSON file where each line is a separate JSON object and converts it into a pandas DataFrame.

#     Parameters:
#     file_path (str): The file path to the JSON file.

#     Returns:
#     pd.DataFrame: A DataFrame where each row corresponds to a JSON object from the file.
#     """
#     # Open the file at the given file path
#     with open(file_path, 'r') as file:
#         # Read each line in the file, parse it as JSON, and add it to a list
#         data = [json.loads(line) for line in file]

#     # Convert the list of dictionaries to a pandas DataFrame
#     df = pd.DataFrame(data)
#     return df
from pydantic import BaseModel, ValidationError
import json
import pandas as pd
from typing import List, Optional

# Define Pydantic models for nested structures
class Customer(BaseModel):
    customerEmail: str
    customerPhone: str
    customerDevice: str
    customerIPAddress: str
    customerBillingAddress: str

class Order(BaseModel):
    orderId: str
    orderAmount: int
    orderState: str
    orderShippingAddress: str

class PaymentMethod(BaseModel):
    paymentMethodId: str
    paymentMethodRegistrationFailure: bool
    paymentMethodType: str
    paymentMethodProvider: str
    paymentMethodIssuer: str

class Transaction(BaseModel):
    transactionId: str
    orderId: str
    paymentMethodId: str
    transactionAmount: int
    transactionFailed: bool

# Main model representing the entire JSON line
class CustomerData(BaseModel):
    fraudulent: bool
    customer: Customer
    orders: List[Order] = []
    paymentMethods: List[PaymentMethod] = []
    transactions: List[Transaction] = []

def read_json_to_dataframe(file_path: str) -> pd.DataFrame:
    """
        Reads a JSON file where each line is a separate JSON object and converts it into a pandas DataFrame.

        Parameters:
        file_path (str): The file path to the JSON file.

        Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a JSON object from the file. We drop rows where any field is an empty dictionary.
     """ 
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Validate each line with the Pydantic model
                parsed_data = CustomerData.parse_raw(line)
                data.append(parsed_data.dict())
            except ValidationError as e:
                # Print error and terminate script
                print(f"Error in line: {line}")
                print(e.json())
                raise SystemExit("Script terminated due to data validation error.")
    df = pd.DataFrame(data)

    # Drop rows where any field is an empty dictionary
    df = df[df[['customer', 'orders', 'paymentMethods', 'transactions']].applymap(lambda x: x != []).all(axis=1)]
    return df


def get_geolocation(ip_address: str) -> dict:
    """
    Fetch geolocation information for a given IP address using the ip-api service.

    Parameters:
    ip_address (str): The IP address for which geolocation information is to be fetched.

    Returns:
    dict: A dictionary containing the country, countryCode, latitude, and longitude.
          Returns None for these fields if the API call fails.
    """
    try:
        response = requests.get(f'http://ip-api.com/json/{ip_address}')
        data = response.json()
        return {
            'country': data.get('country', ''),
            'countryCode': data.get('countryCode', ''),
            'lat': data.get('lat', ''),
            'lon': data.get('lon', '')
        }
    except Exception as e:
        # In case of an exception, return None for all fields
        return {'country': None, 'countryCode': None, 'lat': None, 'lon': None}

def process_customer_ips(df: pd.DataFrame, customer_column: str) -> pd.DataFrame:
    """
    Process a DataFrame to include geolocation information for IP addresses found in a specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing customer data.
    customer_column (str): The name of the column in df that contains customer information
                           in a dictionary format, including the IP address.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for country, countryCode,
                  latitude, and longitude based on the customer's IP address.
    """
    # Extracting customer IP addresses
    df['customerIPAddress'] = df[customer_column].apply(lambda x: x['customerIPAddress'] if 'customerIPAddress' in x else None)

    # Rate limit delay
    rate_limit_delay = 60 / 45  # Adjust based on the API's limit

    # Fetch geolocation data for each IP address
    geo_data = []
    for ip in df['customerIPAddress']:
        if ip:
            geo_info = get_geolocation(ip)
            geo_data.append(geo_info)
            time.sleep(rate_limit_delay)
        else: 
            geo_data.append({'country': None, 'countryCode': None, 'lat': None, 'lon': None})

    # Create a DataFrame from the geolocation data
    geo_df = pd.DataFrame(geo_data)

    # Concatenate the new columns with the original dataframe
    df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

    # Optionally, drop the temporary 'customerIPAddress' column if it's no longer needed
    df.drop('customerIPAddress', axis=1, inplace=True)

    return df

def get_geolocation_from_address(address: str) -> dict:
    """
    Fetch geolocation information for the last part of the billing address using Nominatim (OpenStreetMap) with a free-form query.

    Parameters:
    address (str): The last part of the billing address to geocode.

    Returns:
    dict: A dictionary containing the display name, latitude, and longitude.
    """
    # URL encode the address
    url = f'https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(address)}'
    try:
        response = requests.get(url)
        data = response.json()
        if data and len(data) > 0:
            return {
                'display_name': data[0].get('display_name', ''),
                'lat': data[0].get('lat', ''),
                'lon': data[0].get('lon', '')
            }
        else:
            return {'display_name': None, 'lat': None, 'lon': None}
    except Exception as e:
        print(f"Error fetching geolocation: {e}")
        return {'display_name': None, 'lat': None, 'lon': None}

def process_customer_addresses(df: pd.DataFrame, customer_column: str) -> pd.DataFrame:
    """
    Process a DataFrame to include geolocation information for the last part of addresses found in a specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing customer data.
    customer_column (str): The name of the column in df that contains customer billing addresses.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for display name, latitude, and longitude.
    """
    # Extracting the last part of customerBillingAddress from the customer column
    df['customerBillingAddress'] = df[customer_column].apply(lambda x: x['customerBillingAddress'].split(',')[-1].strip() if 'customerBillingAddress' in x and x['customerBillingAddress'] else None)

    geo_data = []
    for address in df['customerBillingAddress']:
        if address:
            geo_info = get_geolocation_from_address(address)
            geo_data.append(geo_info)
            time.sleep(1)  # Rate limit of one request per second
        else:
            geo_data.append({'display_name': None, 'lat': None, 'lon': None})

    geo_df = pd.DataFrame(geo_data)
    df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)

    return df

def extract_email_domain(email: str) -> str:
    """Extract the domain from an email address."""
    return email.split('@')[-1] if '@' in email else None

def is_free_domain(email_domain: str) -> int:
    """Determine if the email domain is a free domain."""
    free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
    return 1 if email_domain in free_domains else 0

def process_email_domains(df: pd.DataFrame, customer_column: str) -> pd.DataFrame:
    """
    Process a DataFrame to include email domain information and a binary indicator for free vs. corporate domains.

    Parameters:
    df (pd.DataFrame): The DataFrame containing customer data.
    customer_column (str): The name of the column in df that contains dictionaries with customer data.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for email domains and free/corporate domain indicator.
    """
    # Extracting customerEmail from the customer column
    df['customerEmail'] = df[customer_column].apply(lambda x: x['customerEmail'] if 'customerEmail' in x else None)

    # Applying the domain extraction function
    df['email_domain'] = df['customerEmail'].apply(extract_email_domain)
    df['is_free_domain'] = df['email_domain'].apply(is_free_domain)

    return df

def extract_order_features(orders):
    """
    Extracts various features from a list of orders.

    Parameters:
    orders (list): A list of dictionaries, each representing an order.

    Returns:
    dict: A dictionary containing calculated features such as the number of orders,
          average order amount, minimum and maximum order amount, counts of fulfilled,
          failed, and pending orders, and the number of unique shipping addresses.
    """
    # Return default values if there are no orders
    if not orders:
        return {
            'num_orders': 0,
            'avg_order_amount': 0.0,
            'min_order_amount': 0.0,
            'max_order_amount': 0.0,
            'num_fulfilled_orders': 0,
            'num_failed_orders': 0,
            'num_pending_orders': 0,
            'num_unique_shipping_addresses': 0
        }

    # Extract relevant information from each order
    order_amounts = [order['orderAmount'] for order in orders]
    order_states = [order['orderState'] for order in orders]
    shipping_addresses = [order['orderShippingAddress'] for order in orders]

    # Calculate various statistics based on the orders
    num_orders = len(orders)
    avg_order_amount = round(sum(order_amounts) / num_orders, 2) if num_orders else 0.0
    min_order_amount = round(min(order_amounts), 2) if num_orders else 0.0
    max_order_amount = round(max(order_amounts), 2) if num_orders else 0.0
    num_fulfilled_orders = sum(state == 'fulfilled' for state in order_states)
    num_failed_orders = sum(state == 'failed' for state in order_states)
    num_pending_orders = sum(state == 'pending' for state in order_states)
    num_unique_shipping_addresses = len(set(shipping_addresses))

    # Return a dictionary with all the calculated features
    return {
        'num_orders': num_orders,
        'avg_order_amount': avg_order_amount,
        'min_order_amount': min_order_amount,
        'max_order_amount': max_order_amount,
        'num_fulfilled_orders': num_fulfilled_orders,
        'num_failed_orders': num_failed_orders,
        'num_pending_orders': num_pending_orders,
        'num_unique_shipping_addresses': num_unique_shipping_addresses
    }

def process_orders(df, orders_column):
    """
    Processes a DataFrame to include features extracted from each list of orders.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the orders data.
    orders_column (str): The name of the column in df that contains the lists of orders.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for the extracted order features.
    """
    # Apply the extract_order_features function to each list of orders in the DataFrame
    order_features = df[orders_column].apply(extract_order_features)
    # Normalize the features and convert them into a DataFrame
    order_features_df = pd.json_normalize(order_features)
    # Concatenate the original DataFrame with the new features DataFrame
    return pd.concat([df, order_features_df], axis=1)



def process_payment_methods(payment_methods):
    """
    Extracts features from a list of payment methods.

    Parameters:
    payment_methods (list): A list of dictionaries, each representing a payment method.

    Returns:
    dict: A dictionary containing features such as the count of unique issuers,
          registration failure rate, and binary indicators for payment method types and providers.
    """
    # Return default values if there are no payment methods
    if not payment_methods:
        return {}

    # Count occurrences of each payment method type and provider
    method_types = Counter([method['paymentMethodType'] for method in payment_methods])
    providers = Counter([method['paymentMethodProvider'] for method in payment_methods])

    # Get the unique set of payment method issuers
    issuers = set(method['paymentMethodIssuer'] for method in payment_methods)

    # Calculate the registration failure rate
    registration_failures = (sum(method['paymentMethodRegistrationFailure'] for method in payment_methods) / len(payment_methods)) * 100

    # Create binary features for each payment method type and provider
    features = {f'type_{method_type}': int(method_type in method_types) for method_type in all_payment_method_types}
    features.update({f'provider_{provider}': int(provider in providers) for provider in all_payment_method_providers})

    # Add the count of unique issuers and the registration failure rate
    features.update({'unique_issuers': len(issuers), 'registration_failure_rate': round(registration_failures, 2)})

    return features

def preprocess_payment_methods(df, payment_methods_column):
    """
    Processes a DataFrame to include features extracted from each list of payment methods.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the payment methods data.
    payment_methods_column (str): The name of the column in df that contains the lists of payment methods.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for the extracted payment method features.
    """
    global all_payment_method_types, all_payment_method_providers
    all_payment_method_types = set()
    all_payment_method_providers = set()
    
    # Determine all unique payment method types and providers
    for methods in df[payment_methods_column]:
        if methods:
            all_payment_method_types.update(method['paymentMethodType'] for method in methods)
            all_payment_method_providers.update(method['paymentMethodProvider'] for method in methods)
    
    # Apply the feature extraction function to each list of payment methods
    payment_features = df[payment_methods_column].apply(process_payment_methods)
    # Convert the extracted features into a DataFrame
    payment_features_df = pd.DataFrame(payment_features.tolist())

    # Concatenate the original DataFrame with the new features DataFrame
    return pd.concat([df, payment_features_df], axis=1)


def extract_transaction_features(transactions):
    """
    Extracts various features from a list of transactions.

    Parameters:
    transactions (list): A list of dictionaries, each representing a transaction.

    Returns:
    dict: A dictionary containing calculated features such as total, average, minimum, 
          and maximum transaction amounts, the number of transactions, the number of 
          failed transactions, and the failure rate of transactions.
    """
    # Return default values if there are no transactions
    if not transactions:
        return {
            'total_transaction_amount': 0,
            'avg_transaction_amount': 0,
            'min_transaction_amount': 0,
            'max_transaction_amount': 0,
            'num_transactions': 0,
            'num_failed_transactions': 0,
            'failure_rate_transactions': 0
        }

    # Extract relevant information from each transaction
    transaction_amounts = [trans['transactionAmount'] for trans in transactions]
    transaction_failed = [trans['transactionFailed'] for trans in transactions]

    # Calculate various statistics based on the transactions
    num_transactions = len(transactions)
    num_failed_transactions = sum(transaction_failed)

    return {
        'total_transaction_amount': sum(transaction_amounts),
        'avg_transaction_amount': round(sum(transaction_amounts) / num_transactions, 2) if num_transactions else 0,
        'min_transaction_amount': min(transaction_amounts) if num_transactions else 0,
        'max_transaction_amount': max(transaction_amounts) if num_transactions else 0,
        'num_transactions': num_transactions,
        'num_failed_transactions': num_failed_transactions,
        'failure_rate_transactions': round(num_failed_transactions / num_transactions * 100, 2) if num_transactions else 0
    }

def process_transactions(df, transactions_column):
    """
    Processes a DataFrame to include features extracted from each list of transactions.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the transactions data.
    transactions_column (str): The name of the column in df that contains the lists of transactions.

    Returns:
    pd.DataFrame: The original DataFrame with additional columns for the extracted transaction features.
    """
    # Apply the extract_transaction_features function to each list of transactions in the DataFrame
    transaction_features = df[transactions_column].apply(extract_transaction_features)
    # Normalize the features and convert them into a DataFrame
    transaction_features_df = pd.DataFrame(transaction_features.tolist())

    # Concatenate the original DataFrame with the new features DataFrame
    return pd.concat([df, transaction_features_df], axis=1)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the great-circle distance between two points on the Earth's surface.

    Parameters:
    lon1 (float): Longitude of the first point.
    lat1 (float): Latitude of the first point.
    lon2 (float): Longitude of the second point.
    lat2 (float): Latitude of the second point.

    Returns:
    float: The distance between the two points in kilometers.
    """
    # Check if any input values are None and return NaN if so
    if None in [lon1, lat1, lon2, lat2]:
        return np.nan

    # Convert latitude and longitude from decimal degrees to radians
    try:
        lon1, lat1, lon2, lat2 = map(np.radians, map(float, [lon1, lat1, lon2, lat2]))
    except (ValueError, TypeError):
        # Return NaN if values cannot be converted to float
        return np.nan

    # Haversine formula to calculate the distance
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

def add_geographical_distance_feature(df):
    """
    Adds a column to the DataFrame representing the geographical distance
    between two sets of coordinates (IP address and billing address).

    Parameters:
    df (pd.DataFrame): The DataFrame to which the distance feature will be added.

    Returns:
    pd.DataFrame: The DataFrame with the added 'geo_distance' feature.
    """
    # Apply the haversine function to each row in the DataFrame
    df['geo_distance'] = df.apply(lambda x: haversine(x['lonIP'], x['latIP'], x['lon'], x['lat']), axis=1)
    return df

def subset_and_transform_final_dataframe(df):

    df = df[
                [   'fraudulent',
                    'countryIP', 
                    'countryCodeIP', 
                    'latIP', 
                    'lonIP',
                    'lat', 
                    'lon', 
                    'is_free_domain', 
                    'num_orders', 
                    'avg_order_amount',
                    'min_order_amount', 
                    'max_order_amount', 
                    'num_fulfilled_orders',
                    'num_failed_orders', 
                    'num_pending_orders',
                    'num_unique_shipping_addresses', 
                    'type_apple pay', 
                    'type_card',
                    'type_paypal', 
                    'type_bitcoin', 
                    'provider_American Express',
                    'provider_VISA 16 digit', 
                    'provider_VISA 13 digit', 
                    'provider_Maestro',
                    'provider_JCB 15 digit', 
                    'provider_Mastercard', 
                    'provider_Voyager',
                    'provider_Diners Club / Carte Blanche', 
                    'provider_Discover',
                    'provider_JCB 16 digit', 
                    'unique_issuers', 
                    'registration_failure_rate',
                    'total_transaction_amount', 
                    'avg_transaction_amount',
                    'min_transaction_amount', 
                    'max_transaction_amount', 
                    'num_transactions',
                    'num_failed_transactions', 
                    'failure_rate_transactions', 
                    'geo_distance'
                ]
    ]

    df = (
    df.rename(columns={'lat': 'latBillingAddress', 'lon': 'lonBillingAddress'})
      .assign(fraudulent=lambda x: x['fraudulent'].astype(int))
    )
    return df 