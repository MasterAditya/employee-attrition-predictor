# Employee Attrition Predictor 🎯

A machine learning application that predicts employee attrition using HR analytics data. Built with Python, scikit-learn, and Streamlit.

## Features
- Machine learning model trained on HR analytics data
- Interactive web interface for real-time predictions
- Feature importance visualization
- Probability scores and insights
- Clean, modern UI with dark mode support

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)

3. Enter employee details using the sidebar inputs and get instant predictions

## Project Structure
- `app.py`: Streamlit web application
- `model.py`: Model training pipeline
- `predict.py`: Prediction logic and model loading
- `models/`: Saved model and encoders

## Tech Stack
- Python 3.8+
- scikit-learn
- LightGBM
- Streamlit
- Plotly
- SHAP

## Author
Created as a portfolio project for ML Engineering roles.
