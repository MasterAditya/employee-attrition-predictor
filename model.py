import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def create_mock_data(n_samples=2000):
    """Generate synthetic HR data similar to IBM Attrition dataset with enhanced features."""
    np.random.seed(42)
    
    data = {
        'Age': np.random.normal(37, 9, n_samples).astype(int),
        'MonthlyIncome': np.random.normal(6500, 4000, n_samples).astype(int),
        'YearsAtCompany': np.random.normal(7, 6, n_samples).astype(int),
        'DistanceFromHome': np.random.normal(9, 8, n_samples).astype(int),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'PerformanceRating': np.random.randint(1, 5, n_samples),
        'RelationshipSatisfaction': np.random.randint(1, 5, n_samples),
        'EnvironmentSatisfaction': np.random.randint(1, 5, n_samples),
        'TotalWorkingYears': np.random.normal(11, 7, n_samples).astype(int),
        'Department': np.random.choice(['Sales', 'R&D', 'HR'], n_samples),
        'JobRole': np.random.choice(['Sales Executive', 'Research Scientist', 'Manager', 
                                   'HR Representative', 'Developer'], n_samples),
        'EducationField': np.random.choice(['Business', 'Life Sciences', 'Medical', 
                                          'Marketing', 'Technical', 'Other'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'BusinessTravel': np.random.choice(['Rarely', 'Frequently', 'No Travel'], n_samples),
        'OverTime': np.random.choice(['Yes', 'No'], n_samples),
        'StockOptionLevel': np.random.randint(0, 4, n_samples),
    }
    
    # Create balanced attrition rules with targeted weights
    prob_leave = (
        # Primary factors - strong negative indicators
        ((data['JobSatisfaction'] < 2) & (data['MonthlyIncome'] < 3500)) * 0.6 +
        ((data['WorkLifeBalance'] < 2) & (data['OverTime'] == 'Yes')) * 0.55 +
        ((data['PerformanceRating'] < 2) & (data['YearsAtCompany'] > 15)) * 0.5 +
        
        # Secondary factors - moderate indicators
        (data['JobSatisfaction'] < 2) * 0.35 +
        (data['WorkLifeBalance'] < 2) * 0.3 +
        (data['EnvironmentSatisfaction'] < 2) * 0.3 +
        (data['RelationshipSatisfaction'] < 2) * 0.25 +
        (data['MonthlyIncome'] < 3000) * 0.25 +
        
        # Contributing factors - mild indicators
        (data['OverTime'] == 'Yes') * 0.15 +
        (data['BusinessTravel'] == 'Frequently') * 0.15 +
        (data['StockOptionLevel'] == 0) * 0.1 +
        (data['Age'] < 25) * 0.1
    )
    
    # Normalize probabilities with moderate cap
    prob_leave = prob_leave / prob_leave.max() * 0.8  # Cap at 80% probability
    data['Attrition'] = np.random.binomial(1, prob_leave)
    
    return pd.DataFrame(data)

def prepare_data(df):
    """Prepare data for model training."""
    # Create copy to avoid modifying original data
    df = df.copy()
    
    # Create label encoders for categorical columns
    categorical_cols = ['Department', 'JobRole', 'EducationField', 'MaritalStatus',
                       'BusinessTravel', 'OverTime']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    
    # Scale numerical columns
    numerical_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'DistanceFromHome',
                     'TotalWorkingYears', 'StockOptionLevel', 'JobSatisfaction',
                     'WorkLifeBalance', 'PerformanceRating', 'RelationshipSatisfaction',
                     'EnvironmentSatisfaction']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, encoders, scaler

def train_model():
    """Train the attrition prediction model."""
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate mock data
    df = create_mock_data()
    
    # Prepare features
    df_processed, encoders, scaler = prepare_data(df)
    
    # Split features and target
    X = df_processed.drop('Attrition', axis=1)
    y = df_processed['Attrition']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with balanced parameters
    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=7,
        num_leaves=40,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=2,  # Give more weight to positive class
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Save model and preprocessors
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(encoders, 'models/encoders.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return metrics

if __name__ == '__main__':
    metrics = train_model()
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
