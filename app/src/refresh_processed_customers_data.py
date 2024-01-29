# Imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import logging
from utils import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def main():
    logger.info("Starting the data processing script.")

    # Read the JSON file into a pandas DataFrame
    file_path = '../../raw_data_files/customers.json'
    df = read_json_to_dataframe(file_path)
    logger.info("Read JSON data into DataFrame.")

    # Process customer IPs
    df = process_customer_ips(df, 'customer')
    df.columns = ['fraudulent', 'customer', 'orders', 'paymentMethods', 'transactions',
                  'countryIP', 'countryCodeIP', 'latIP', 'lonIP']
    logger.info("Processed customer IPs.")

    # Process customer billing addresses
    df = process_customer_addresses(df, 'customer')
    logger.info("Processed customer billing addresses.")

    # Process customer device info
    df = process_email_domains(df, 'customer')
    logger.info("Processed customer email domains.")

    # Process orders
    df = process_orders(df, 'orders')
    logger.info("Processed orders.")

    # Process payment methods
    df = preprocess_payment_methods(df, 'paymentMethods')
    logger.info("Processed payment methods.")

    # Process transactions
    df = process_transactions(df, 'transactions')
    logger.info("Processed transactions.")

    # Process customers IP and billing address's IP distance
    df = add_geographical_distance_feature(df)
    logger.info("Calculated geographical distances between IPs and billing addresses.")

    # final transofrmation of the dataframe
    df = subset_and_transform_final_dataframe(df)
    logger.info("Final changes to the DataFrame applied.")

    # Save the final DataFrame as a CSV file in the prepared_data_files folder
    output_file_path = '../../prepared_data_files/processed_customers.csv'
    df.to_csv(output_file_path, index=False)
    logger.info(f"Saved the processed DataFrame to {output_file_path}.")

    logger.info("Data processing completed.")
    return None

if __name__ == "__main__":
    main()
