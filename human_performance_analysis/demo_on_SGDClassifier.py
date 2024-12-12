import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def generate_data(n_samples=100, seed=0):
    """
    Generate synthetic time series data with binary features.
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Generated dataset
    """
    np.random.seed(seed)
    
    # Create date range
    dates = pd.date_range(start='2021-01-01', periods=n_samples, freq='D')
    
    # Generate features with some controlled randomness
    def generate_binary_feature(p_one=0.5):
        return np.random.choice([0, 1], size=n_samples, p=[1-p_one, p_one])
    
    # Create DataFrame
    data = pd.DataFrame({
        'Date': dates,
        'Kismet': generate_binary_feature(0.6),
        'Psyche': generate_binary_feature(0.4),
        'Soma': generate_binary_feature(0.5),
        'Pneuma': generate_binary_feature(0.7),
        'Opus': generate_binary_feature(0.3),
        'Notes': ['Note'] * n_samples
    })
    
    return data

def prepare_data(data):
    """
    Prepare data for online learning by creating cumulative features.
    
    Args:
        data (pandas.DataFrame): Input dataset
    
    Returns:
        tuple: Scaled features, target, and original data
    """
    # Compute cumulative sums
    domains = ['Kismet', 'Psyche', 'Soma', 'Pneuma', 'Opus']
    for domain in domains:
        data[f'{domain}_cum'] = data[domain].cumsum()
    
    # Prepare features and target
    features = [f'{domain}_cum' for domain in domains]
    X = data[features].values
    
    # Randomly choose a domain to predict
    predict_domain = random.choice(domains)
    print(f"Predicting domain: {predict_domain}")
    y = data[predict_domain].values
    
    # Prepare other features for prediction
    feature_domains = [d for d in domains if d != predict_domain]
    X_features = data[[f'{d}_cum' for d in feature_domains]].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    
    return X_scaled, y, data, predict_domain

def online_learning(X, y, test_size=0.2):
    """
    Perform online learning using Stochastic Gradient Descent Classifier.
    
    Args:
        X (numpy.ndarray): Scaled features
        y (numpy.ndarray): Target variable
        test_size (float): Proportion of data to use for testing
    
    Returns:
        tuple: Predicted values, accuracy score
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Initialize online learning classifier
    clf = SGDClassifier(
        loss='log_loss',  # Logistic regression loss
        penalty='l2',     # L2 regularization
        max_iter=1000,    # Maximum number of iterations
        learning_rate='constant',  # Constant learning rate
        eta0=0.01,        # Learning rate
        random_state=42
    )
    
    # Initialize prediction and true values lists
    y_pred_list = []
    
    # Incremental learning
    for i in range(len(X_train)):
        # Partial fit with current sample
        clf.partial_fit(X_train[i:i+1], y_train[i:i+1], classes=np.unique(y))
        
        # Predict and store
        if i > 0:
            y_pred_list.append(clf.predict(X_train[i:i+1])[0])
    
    # Predict on test set
    y_test_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    return y_pred_list, test_accuracy

def plot_results(data, y_pred, test_accuracy, predict_domain):
    """
    Create visualizations of the online learning results.
    
    Args:
        data (pandas.DataFrame): Original dataset
        y_pred (list): Predicted values
        test_accuracy (float): Accuracy on test set
        predict_domain (str): Domain being predicted
    """
    # Cumulative plot
    plt.figure(figsize=(15, 10))
    domains = ['Kismet', 'Psyche', 'Soma', 'Pneuma', 'Opus']
    
    # Plot actual cumulative for all domains
    for domain in domains:
        plt.plot(data['Date'], data[f'{domain}_cum'], label=f'{domain} (Cumulative)', marker='o')
    
    # Create predicted cumulative line
    predicted_cum = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        predicted_cum[i] = predicted_cum[i-1] + y_pred[i] if i > 0 else y_pred[i]
    
    # Plot predicted cumulative line for the selected domain
    plt.plot(data['Date'][:len(predicted_cum)], predicted_cum, 
             label=f'{predict_domain} Predicted (Cumulative)', 
             marker='x', linestyle='--', color='red')
    
    plt.title(f'Cumulative Plot (Predicting {predict_domain}, Test Accuracy: {test_accuracy:.2%})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Prediction comparison
    plt.figure(figsize=(15, 6))
    plt.plot(data['Date'][:len(y_pred)], data[predict_domain][:len(y_pred)], 
             label=f'Actual {predict_domain}', marker='o')
    plt.plot(data['Date'][:len(y_pred)], y_pred, 
             label=f'Predicted {predict_domain}', marker='x', linestyle='--')
    plt.title(f'Prediction Comparison (Test Accuracy: {test_accuracy:.2%})')
    plt.xlabel('Date')
    plt.ylabel(f'{predict_domain} Value')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to orchestrate data generation, online learning, and visualization.
    """
    # Generate synthetic data
    data = generate_data()
    
    # Prepare data
    X_scaled, y, processed_data, predict_domain = prepare_data(data)
    
    # Perform online learning
    y_pred, test_accuracy = online_learning(X_scaled, y)
    
    # Visualize results
    plot_results(processed_data, y_pred, test_accuracy, predict_domain)

if __name__ == "__main__":
    main()

"""

"""