# Imports
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Load the data
    data = pd.read_csv('../../prepared_data_files/processed_customers.csv')
    data = data.drop(['countryIP', 'countryCodeIP', 'latIP', 'lonIP', 'latBillingAddress', 'lonBillingAddress', 'geo_distance'], axis=1)
    
    # Prepare X and y
    X = data.drop('fraudulent', axis=1)
    y = data['fraudulent']
    
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Random Forest with Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'class_weight': [None, 'balanced']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_smote, y_train_smote)
    
    # Save the best estimator
    best_rf = grid_search.best_estimator_
    with open('../../model/random_forest_latest_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)
    
    # Log the success
    logging.info("Random Forest model trained and saved successfully.")

if __name__ == "__main__":
    main()
