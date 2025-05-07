import streamlit as st
import plotly.graph_objects as go
from predict import AttritionPredictor
import os

# Page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stAlert {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .stButton>button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    .plot-container {
        border-radius: 1rem;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def create_gauge_chart(probability):
    """Create a gauge chart for probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': "Attrition Probability (%)"}
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(importance_dict):
    """Create a horizontal bar chart for feature importance."""
    fig = go.Figure(go.Bar(
        x=list(importance_dict.values()),
        y=list(importance_dict.keys()),
        orientation='h',
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title="Top 5 Important Features",
        xaxis_title="Feature Importance (SHAP value)",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Title and description
    st.title("🎯 Employee Attrition Predictor")
    st.markdown("""
        <div style='background-color: rgba(255,255,255,0.05); padding: 1rem; border-radius: 0.75rem; margin-bottom: 2rem;'>
        Predict employee attrition risk using advanced machine learning. This tool analyzes various factors
        to help identify employees who might be at risk of leaving the organization.
        </div>
    """, unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('models/model.joblib'):
        st.error("Model not found. Please run model.py first to train the model.")
        return
    
    # Initialize predictor
    predictor = AttritionPredictor()
    
    # Organize inputs into tabs in sidebar
    st.sidebar.header("📝 Employee Information")
    
    tab1, tab2, tab3 = st.sidebar.tabs(["Personal", "Job", "Satisfaction"])
    
    with tab1:
        st.subheader("Personal Details")
        age = st.slider("🎂 Age", 18, 65, 30)
        marital_status = st.selectbox(
            "💑 Marital Status",
            ["Single", "Married", "Divorced"]
        )
        distance = st.slider("🏠 Distance From Home (miles)", 1, 30, 5)
        business_travel = st.selectbox(
            "✈️ Business Travel",
            ["Rarely", "Frequently", "No Travel"]
        )
    
    with tab2:
        st.subheader("Job Details")
        department = st.selectbox(
            "🏢 Department",
            ["Sales", "R&D", "HR"]
        )
        job_role = st.selectbox(
            "💼 Job Role",
            ["Sales Executive", "Research Scientist", "Manager", "HR Representative", "Developer"]
        )
        years = st.slider("📅 Years at Company", 0, 40, 5)
        total_years = st.slider("⌛ Total Working Years", 0, 45, 8)
        monthly_income = st.slider("💰 Monthly Income ($)", 2000, 20000, 6000)
        stock_level = st.slider("📈 Stock Option Level", 0, 3, 0)
        overtime = st.selectbox(
            "⏰ Overtime",
            ["No", "Yes"]
        )
    
    with tab3:
        st.subheader("Satisfaction Metrics")
        education_field = st.selectbox(
            "🎓 Education Field",
            ["Business", "Life Sciences", "Medical", "Marketing", "Technical", "Other"]
        )
        job_satisfaction = st.slider("😊 Job Satisfaction", 1, 4, 3, help="1=Low, 4=High")
        work_life_balance = st.slider("⚖️ Work Life Balance", 1, 4, 3, help="1=Bad, 4=Best")
        performance_rating = st.slider("⭐ Performance Rating", 1, 4, 3, help="1=Low, 4=High")
        relationship_satisfaction = st.slider("👥 Relationship Satisfaction", 1, 4, 3, help="1=Low, 4=High")
        environment_satisfaction = st.slider("🌟 Environment Satisfaction", 1, 4, 3, help="1=Low, 4=High")
    
    # Create input data dictionary with new features
    input_data = {
        'Age': age,
        'MaritalStatus': marital_status,
        'DistanceFromHome': distance,
        'Department': department,
        'JobRole': job_role,
        'YearsAtCompany': years,
        'TotalWorkingYears': total_years,
        'MonthlyIncome': monthly_income,
        'EducationField': education_field,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'EnvironmentSatisfaction': environment_satisfaction,
        'BusinessTravel': business_travel,
        'OverTime': overtime,
        'StockOptionLevel': stock_level
    }
    
    # Make prediction when user clicks the button
    if st.sidebar.button("Predict Attrition"):
        # Get prediction and explanation
        result = predictor.predict(input_data)
        importance = predictor.explain_prediction(input_data)
        
        # Create two columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            # Display prediction result with enhanced styling
            if result['prediction'] == 1:
                st.error("""
                    ### ⚠️ High Risk of Attrition Detected!
                    This employee shows significant attrition risk factors that require immediate attention.
                    Consider scheduling a retention interview and reviewing their engagement factors.
                """)
            else:
                st.success("""
                    ### ✅ Employee Likely to Stay
                    This employee shows strong retention indicators and appears well-engaged.
                    Continue supporting their growth and maintaining positive workplace factors.
                """)
            
            # Display gauge chart
            st.plotly_chart(create_gauge_chart(result['probability']))
            
        with col2:
            # Display feature importance
            st.plotly_chart(create_feature_importance_chart(importance))
            
        # Additional insights
        st.subheader("📊 Detailed Analysis")
        st.write(f"""
        - Attrition Probability: {result['probability']:.1%}
        - Confidence Level: {'High' if abs(result['probability'] - 0.5) > 0.3 else 'Moderate'}
        """)

if __name__ == "__main__":
    main()
