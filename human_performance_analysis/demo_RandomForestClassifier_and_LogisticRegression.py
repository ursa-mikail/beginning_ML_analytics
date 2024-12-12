import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import random

class AdvancedTimeSeriesPrediction:
    def __init__(self, n_samples=100, seed=42):
        """
        Initialize the time series prediction experiment
        
        Args:
            n_samples (int): Number of samples to generate
            seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        self.domains = ['Kismet', 'Psyche', 'Soma', 'Pneuma', 'Opus']
        self.data = self.generate_data(n_samples)
        self.prediction_results = {}
    
    def generate_data(self, n_samples):
        """
        Generate synthetic time series data with complex interactions
        
        Args:
            n_samples (int): Number of samples to generate
        
        Returns:
            pandas.DataFrame: Generated dataset
        """
        dates = pd.date_range(start='2021-01-01', periods=n_samples, freq='D')
        
        def generate_dependent_feature(dependencies, noise_level=0.2):
            """
            Generate a feature based on dependencies with other features
            
            Args:
                dependencies (dict): Dictionary of {domain: weight}
                noise_level (float): Level of random noise to add
            
            Returns:
                numpy.ndarray: Generated feature
            """
            feature = np.zeros(n_samples)
            for domain, weight in dependencies.items():
                feature += self.data[domain] * weight if 'data' in locals() else 0
            
            # Add some randomness
            feature += np.random.normal(0, noise_level, n_samples)
            
            # Threshold to binary
            return (feature > np.median(feature)).astype(int)
        
        # Create initial random binary features
        data = pd.DataFrame({
            'Date': dates,
            'Kismet': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
            'Psyche': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
            'Soma': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
            'Pneuma': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
            'Opus': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        })
        
        # Add cumulative and lagged features
        for domain in self.domains:
            data[f'{domain}_cum'] = data[domain].cumsum()
            data[f'{domain}_lag1'] = data[domain].shift(1).fillna(0)
            data[f'{domain}_lag2'] = data[domain].shift(2).fillna(0)
        
        # Add interaction features
        data['complex_interaction'] = generate_dependent_feature({
            'Kismet': 0.5, 
            'Psyche': 0.3, 
            'Soma': -0.2
        })
        
        return data
    
    def prepare_data(self, predict_domain='Kismet'):
        """
        Prepare data for machine learning
        
        Args:
            predict_domain (str): Domain to predict
        
        Returns:
            tuple: Prepared features, target, and metadata
        """
        # Select all cumulative and lagged features except for the target domain
        feature_columns = [
            f'{domain}_cum' for domain in self.domains if domain != predict_domain
        ] + [
            f'{domain}_lag1' for domain in self.domains if domain != predict_domain
        ] + [
            f'{domain}_lag2' for domain in self.domains if domain != predict_domain
        ] + ['complex_interaction']
        
        X = self.data[feature_columns].values
        y = self.data[predict_domain].values
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def evaluate_models(self, predict_domain='Kismet'):
        """
        Evaluate multiple machine learning models
        
        Args:
            predict_domain (str): Domain to predict
        
        Returns:
            dict: Performance of different models
        """
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(predict_domain)
        
        # Models to try
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'SVM': SVC(probability=True)
        }
        
        results = {}
        for name, model in models.items():
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred),
                'model': model
            }
            
            print(f"\n{name} for predicting {predict_domain}:")
            print(f"Accuracy: {accuracy:.2%}")
            print("Classification Report:")
            print(results[name]['report'])
        
        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        
        # Store prediction results
        best_model = results[best_model_name]['model']
        full_X = self.data[feature_columns].values
        scaler = StandardScaler()
        full_X_scaled = scaler.fit_transform(full_X)
        full_y_pred = best_model.predict(full_X_scaled)
        
        self.prediction_results = {
            'predict_domain': predict_domain,
            'best_model': best_model_name,
            'y_pred': full_y_pred,
            'feature_columns': feature_columns
        }
        
        return results
    
    def plot_results(self):
        """
        Visualize prediction results
        """
        if not self.prediction_results:
            print("No prediction results to plot. Run evaluate_models() first.")
            return
        
        predict_domain = self.prediction_results['predict_domain']
        y_pred = self.prediction_results['y_pred']
        best_model = self.prediction_results['best_model']
        
        # Cumulative plot
        plt.figure(figsize=(15, 10))
        
        # Plot actual cumulative for all domains
        for domain in self.domains:
            plt.plot(self.data['Date'], self.data[f'{domain}_cum'], 
                     label=f'{domain} (Actual Cumulative)', marker='o')
        
        # Create predicted cumulative line
        predicted_cum = np.zeros_like(y_pred, dtype=float)
        for i in range(len(y_pred)):
            predicted_cum[i] = predicted_cum[i-1] + y_pred[i] if i > 0 else y_pred[i]
        
        # Plot predicted cumulative line
        plt.plot(self.data['Date'][:len(predicted_cum)], predicted_cum, 
                 label=f'{predict_domain} Predicted (Cumulative)', 
                 marker='x', linestyle='--', color='red')
        
        plt.title(f'Cumulative Plot (Predicting {predict_domain} with {best_model})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Original vs Predicted plot
        plt.figure(figsize=(15, 6))
        plt.plot(self.data['Date'][:len(y_pred)], 
                 self.data[predict_domain][:len(y_pred)], 
                 label=f'Actual {predict_domain}', marker='o')
        plt.plot(self.data['Date'][:len(y_pred)], y_pred, 
                 label=f'Predicted {predict_domain}', marker='x', linestyle='--')
        plt.title(f'Prediction Comparison for {predict_domain} ({best_model})')
        plt.xlabel('Date')
        plt.ylabel(f'{predict_domain} Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    # Create prediction experiment
    experiment = AdvancedTimeSeriesPrediction()
    
    # Randomly select a domain to predict
    predict_domain = random.choice(experiment.domains)
    print(f"Predicting domain: {predict_domain}")
    
    # Evaluate models
    experiment.evaluate_models(predict_domain)
    
    # Plot results
    experiment.plot_results()

if __name__ == "__main__":
    main()

"""
Predicting domain: Pneuma

Logistic Regression for predicting Pneuma:
Accuracy: 65.00%        // as this is on data that is randomly generated
Classification Report:
              precision    recall  f1-score   support

           0       0.73      0.67      0.70        12
           1       0.56      0.62      0.59         8

    accuracy                           0.65        20
   macro avg       0.64      0.65      0.64        20
weighted avg       0.66      0.65      0.65        20


Random Forest for predicting Pneuma:
Accuracy: 65.00%
Classification Report:
              precision    recall  f1-score   support

           0       0.69      0.75      0.72        12
           1       0.57      0.50      0.53         8

    accuracy                           0.65        20
   macro avg       0.63      0.62      0.63        20
weighted avg       0.64      0.65      0.65        20


SVM for predicting Pneuma:
Accuracy: 75.00%
Classification Report:
              precision    recall  f1-score   support

           0       0.73      0.92      0.81        12
           1       0.80      0.50      0.62         8

    accuracy                           0.75        20
   macro avg       0.77      0.71      0.72        20
weighted avg       0.76      0.75      0.74        20
"""