# Fraud detection challenge

### Main objective

In the "raw_data_files" folder there are some json profiles representing fictional customers from an ecommerce company. The profiles contain information about the customer, their orders, their transactions, what payment methods they used and whether the customer is fraudulent or not. The task is to:

- Transform the json profiles into a dataframe of feature vectors.
- Provide exploratory analysis of the dataset, and to summarise and explain the key trends in the data, explaining which factors appear to be most important in predicting fraud.
- Construct a model to predict if a customer is fraudulent based on their profile.
- Report on the models success and show what features are most important in that model.

### Project Structure

```
ravellintechtest
│   requirements.txt
│   README.md
│   .gitignore
└───app
│   └──src
│   │   └──__init__.py
│   │   └──main.py
│   │   └──utils.py
│   │   └──refresh_processed_customers_data.py
│   └──test
│   │   └──__init__.py
│   │   └──test_refresh_processed_customers_data.py
└───ExploratoryAnalysis
│   │   DataExploration.ipynb
│   │   ModelExploration.ipynb
└───model
│   │   random_forest_latest_model.pkl
└───prepared_data_files
│   │   processed_customers.csv
└───raw_data_files
│   │   customers.json
```
### Feature engineering 

The customers.json file is loaded into a DataFrame, with each field of the JSON (customer, orders, paymentMethods, transactions) represented as columns containing dictionaries. Since the model aims to predict whether a customer is fraudulent based on their profile, we process each column separately to create aggregated fields for each customer. The 'ip-api' API is utilized to fetch geolocation information for the IP addresses of all customers. Additionally, the 'Nominatim' API, with a free-form query, is used to obtain geolocation data for the last part (after the comma) of each customer's billing address. Finally, we exclude customers who have at least one empty dictionary in their respective fields, as our objective is to retain only those customers with comprehensive profile information.

### What are the main steps

Clone this repository, create a virtual environment, and install the dependencies listed in requirements.txt. Then, navigate to the src directory and execute the following command to run the preprocessing script. This script will process the data and save the resulting DataFrame in the prepared_data_files folder:

```bash
python3 refresh_processed_customers_data.py
```

Next, navigate to the 'ExploratoryAnalysis' folder and review the 'DataExploration' and 'ModelExploration' files. These files address the main research questions of this challenge.

Finally, return to the src directory and execute the specified command to run the script that runs and saves the best model as a pickle file. This file will be stored in the 'model' folder, allowing you to test the model's performance on unseen data.

```bash
python3 main.py
```

## Running the tests
Nagivate to the test directory and run the following command (more tests to be added):

```bash
pytest test_refresh_processed_customers_data.py
```
## Author

Vasilis Panagaris