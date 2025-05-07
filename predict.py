import joblib
import pandas as pd
import numpy as np
import shap

class AttritionPredictor:
    def __init__(self):
        """Load the trained model and preprocessors."""
        self.model = joblib.load('models/model.joblib')
        self.encoders = joblib.load('models/encoders.joblib')
        self.scaler = joblib.load('models/scaler.joblib')
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction."""
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        # Create copy to avoid modifying original data
        df = input_data.copy()
        
        # Encode categorical variables
        categorical_cols = ['Department', 'JobRole', 'EducationField', 'MaritalStatus',
                          'BusinessTravel', 'OverTime']
        for col in categorical_cols:
            df[col] = self.encoders[col].transform(df[col])
        
        # Scale numerical variables
        numerical_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
                         'TotalWorkingYears', 'StockOptionLevel', 'JobSatisfaction',
                         'WorkLifeBalance', 'PerformanceRating', 'RelationshipSatisfaction',
                         'EnvironmentSatisfaction']
        df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def predict(self, input_data):
        """Make prediction and return probability."""
        processed_data = self.preprocess_input(input_data)
        probability = self.model.predict_proba(processed_data)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': probability
        }
    
    def explain_prediction(self, input_data):
        """Get SHAP values for feature importance."""
        processed_data = self.preprocess_input(input_data)
        shap_values = self.explainer.shap_values(processed_data)
        
        # Get feature importance
        if isinstance(shap_values, list):  # For tree models
            shap_values = shap_values[1]  # Get values for positive class
            
        feature_importance = dict(zip(processed_data.columns, np.abs(shap_values[0])))
        sorted_importance = dict(sorted(feature_importance.items(), 
                                     key=lambda x: abs(x[1]), 
                                     reverse=True)[:5])
        
        return sorted_importance
