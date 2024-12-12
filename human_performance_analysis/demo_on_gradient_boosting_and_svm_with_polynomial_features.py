import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import random

class EnhancedTimeSeriesPrediction:
    def __init__(self, n_samples=100, seed=42):
        """
        Enhanced time series prediction experiment with advanced techniques
        
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
        
        def generate_nonlinear_feature(dependencies, noise_level=0.2):
            """
            Generate a feature with nonlinear dependencies
            
            Args:
                dependencies (dict): Dictionary of {domain: weight}
                noise_level (float): Level of random noise to add
            
            Returns:
                numpy.ndarray: Generated feature
            """
            feature = np.zeros(n_samples)
            for domain, weight in dependencies.items():
                # Add nonlinear interaction
                feature += np.power(self.data[domain], 2) * weight if 'data' in locals() else 0
            
            # Add more complex noise
            feature += np.random.normal(0, noise_level, n_samples) * np.sin(np.arange(n_samples))
            
            # Threshold to binary with more complex logic
            return (feature > np.percentile(feature, 60)).astype(int)
        
        # Create initial random binary features with more varied distributions
        data = pd.DataFrame({
            'Date': dates,
            'Kismet': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
            'Psyche': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
            'Soma': np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
            'Pneuma': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
            'Opus': np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]),
        })
        
        # Add more sophisticated features
        for domain in self.domains:
            # Cumulative with exponential decay
            data[f'{domain}_cum'] = data[domain].cumsum() * np.exp(-0.05 * np.arange(n_samples))
            data[f'{domain}_lag1'] = data[domain].shift(1).fillna(data[domain].mean())
            data[f'{domain}_lag2'] = data[domain].shift(2).fillna(data[domain].mean())
            data[f'{domain}_rolling_mean'] = data[domain].rolling(window=3, min_periods=1).mean()
        
        # Add more complex interaction features
        data['complex_interaction'] = generate_nonlinear_feature({
            'Kismet': 0.5, 
            'Psyche': 0.3, 
            'Soma': -0.2
        })
        
        return data
    
    def prepare_data(self, predict_domain='Kismet'):
        """
        Prepare data for machine learning with advanced feature engineering
        
        Args:
            predict_domain (str): Domain to predict
        
        Returns:
            tuple: Prepared features, target, and metadata
        """
        # Expanded feature set
        feature_columns = [
            f'{domain}_cum' for domain in self.domains if domain != predict_domain
        ] + [
            f'{domain}_lag1' for domain in self.domains if domain != predict_domain
        ] + [
            f'{domain}_lag2' for domain in self.domains if domain != predict_domain
        ] + [
            f'{domain}_rolling_mean' for domain in self.domains if domain != predict_domain
        ] + ['complex_interaction']
        
        X = self.data[feature_columns].values
        y = self.data[predict_domain].values
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def evaluate_models(self, predict_domain='Kismet'):
        """
        Evaluate multiple machine learning models with advanced techniques
        
        Args:
            predict_domain (str): Domain to predict
        
        Returns:
            dict: Performance of different models
        """
        # Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data(predict_domain)
        
        # Advanced pipelines with feature selection and hyperparameter tuning
        pipelines = {
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=5)),  # Feature selection
                ('classifier', GradientBoostingClassifier())
            ]),
            'SVM with Polynomial Features': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True))
            ])
        }
        
        # Hyperparameter grids
        param_grids = {
            'Gradient Boosting': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.5],
                'classifier__max_depth': [3, 5, 7]
            },
            'SVM with Polynomial Features': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['poly', 'rbf']
            }
        }
        
        results = {}
        for name, pipeline in pipelines.items():
            # Grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[name], 
                cv=5, 
                scoring='accuracy', 
                n_jobs=-1
            )
            
            # Fit and predict
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Compute performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'report': classification_report(y_test, y_pred),
                'best_params': grid_search.best_params_,
                'model': best_model
            }
            
            print(f"\n{name} for predicting {predict_domain}:")
            print(f"Accuracy: {accuracy:.2%}")
            print("Best Parameters:", grid_search.best_params_)
            print("Classification Report:")
            print(results[name]['report'])
        
        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']
        
        # Predict full dataset
        scaler = StandardScaler()
        full_X = self.data[feature_columns].values
        full_y = self.data[predict_domain].values
        
        # Store prediction results
        self.prediction_results = {
            'predict_domain': predict_domain,
            'best_model': best_model_name,
            'y_true': full_y,
            'y_pred': best_model.predict(full_X),
            'feature_columns': feature_columns
        }
        
        return results
    
    def plot_results(self):
        """
        Visualize prediction results with more detailed insights
        """
        if not self.prediction_results:
            print("No prediction results to plot. Run evaluate_models() first.")
            return
        
        predict_domain = self.prediction_results['predict_domain']
        y_true = self.prediction_results['y_true']
        y_pred = self.prediction_results['y_pred']
        best_model = self.prediction_results['best_model']
        
        # Confusion matrix visualization
        plt.figure(figsize=(15, 5))
        
        # Actual vs Predicted
        plt.subplot(121)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.title(f'Actual vs Predicted {predict_domain}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        
        # Prediction Distribution
        plt.subplot(122)
        plt.hist([y_true, y_pred], label=['Actual', 'Predicted'], alpha=0.7, bins=2)
        plt.title(f'Distribution of {predict_domain}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create enhanced prediction experiment
    experiment = EnhancedTimeSeriesPrediction(n_samples=200)  # Increased samples
    
    # Randomly select a domain to predict
    predict_domain = random.choice(experiment.domains)
    print(f"Predicting domain: {predict_domain}")
    
    # Evaluate models with advanced techniques
    experiment.evaluate_models(predict_domain)
    
    # Plot detailed results
    experiment.plot_results()

if __name__ == "__main__":
    main()

"""
Predicting domain: Psyche

Gradient Boosting for predicting Psyche:
Accuracy: 52.50%        // on data produced randomly
Best Parameters: {'classifier__learning_rate': 0.01, 'classifier__max_depth': 3, 'classifier__n_estimators': 100}
Classification Report:
              precision    recall  f1-score   support

           0       0.25      0.13      0.17        15
           1       0.59      0.76      0.67        25

    accuracy                           0.53        40
   macro avg       0.42      0.45      0.42        40
weighted avg       0.46      0.53      0.48        40


SVM with Polynomial Features for predicting Psyche:
Accuracy: 50.00%
Best Parameters: {'classifier__C': 1, 'classifier__kernel': 'rbf'}
Classification Report:
              precision    recall  f1-score   support

           0       0.22      0.13      0.17        15
           1       0.58      0.72      0.64        25

    accuracy                           0.50        40
   macro avg       0.40      0.43      0.40        40
weighted avg       0.45      0.50      0.46        40
"""