# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student At-Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .at-risk {
        background-color: #ffcccc;
        border: 2px solid #ff4444;
    }
    .not-at-risk {
        background-color: #ccffcc;
        border: 2px solid #44ff44;
    }
    .feature-value {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        # Load the dataset
        df = pd.read_csv('Balanced_Realistic_Student_Dataset_v2.csv')
        
        # Create a clean copy
        df_clean = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = [
            'Timestamp',
            'Year Level (last semester)',
            'College (last semester)',
            'Sex',
            'How many subjects have you failed in the most recent semester?',
            'What was your General Weighted Average (GWA) for the most recent semester?',
            'Approximately, what was your attendance rate in classes last semester?'
        ]
        
        existing_columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        df_clean = df_clean.drop(columns=existing_columns_to_drop)
        
        # Rename columns for convenience
        rename_map = {
            'On average, how many hours did you spend gaming per day?': 'Hours Gaming',
            'How often did you submit school requirements(e.g. Assignments, Projects) late?': 'Late Submissions',
            'On average, how many hours did you study on weekdays?': 'Study Hours (Weekdays)',
            'On average, how many hours did you study on weekends?': 'Study Hours (Weekends)',
            'If you answered "Yes" to the question above, how many hours did you work per week?': 'Work Hours',
            'If you answered "Yes" to the question above, how many hours per week did you spend on extracurricular activities in the last semester?': 'Hours Extracurricular Activity',
            'How many academic units were you enrolled in during the most recent (last) semester?': 'Academic unit',
            'Were you involved in extracurricular activities (e.g., student orgs, sports, volunteering)?': 'Extracurricular Activities',
            'Stress level': 'Stress Level',
            'Level of Social Support': 'Social Support',
            'On average, how many hours did you sleep per night?': 'Sleep Hours'
        }
        
        existing_columns_to_rename = {old_name: new_name for old_name, new_name in rename_map.items() if old_name in df_clean.columns}
        df_clean.rename(columns=existing_columns_to_rename, inplace=True)
        
        # Clean numeric columns
        def clean_numeric_column(column_name, default_value=0):
            if column_name in df_clean.columns:
                df_clean[column_name] = df_clean[column_name].replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], np.nan)
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
                if df_clean[column_name].isna().any():
                    fill_value = df_clean[column_name].median() if not df_clean[column_name].isna().all() else default_value
                    df_clean[column_name].fillna(fill_value, inplace=True)
            return True
        
        # Clean specific columns
        df_clean['Hours Gaming'] = df_clean['Hours Gaming'].astype(str)
        df_clean['Hours Gaming'] = df_clean['Hours Gaming'].replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], '0')
        df_clean['Hours Gaming'] = pd.to_numeric(df_clean['Hours Gaming'], errors='coerce')
        df_clean['Hours Gaming'].fillna(0, inplace=True)
        
        clean_numeric_column('Work Hours', 0)
        clean_numeric_column('Hours Extracurricular Activity', 0)
        clean_numeric_column('Study Hours (Weekdays)', 0)
        clean_numeric_column('Study Hours (Weekends)', 0)
        clean_numeric_column('Academic unit', 0)
        clean_numeric_column('Sleep Hours', 7)
        clean_numeric_column('Stress Level', 3)
        clean_numeric_column('Social Support', 3)
        clean_numeric_column('Financial Difficulty', 3)
        
        # Encode Late Submissions
        if 'Late Submissions' in df_clean.columns and df_clean['Late Submissions'].dtype == 'object':
            late_mapping = {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4}
            df_clean['Late Submissions'] = df_clean['Late Submissions'].map(late_mapping)
            df_clean['Late Submissions'].fillna(2.5, inplace=True)
        elif 'Late Submissions' in df_clean.columns:
            df_clean['Late Submissions'] = df_clean['Late Submissions'].replace(0, 1)
        
        # Create engineered features
        df_clean["Total Study Hours"] = df_clean["Study Hours (Weekdays)"] + df_clean["Study Hours (Weekends)"]
        df_clean["StudyEfficiency"] = df_clean["Total Study Hours"] / (df_clean["Late Submissions"] + 0.1)
        df_clean["AcademicEngagement"] = df_clean["Hours Extracurricular Activity"] + df_clean["Social Support"]
        df_clean["StressBalance"] = df_clean["Stress Level"] - df_clean["Social Support"]
        df_clean["TimeBurden"] = df_clean["Work Hours"] + df_clean["Hours Gaming"]
        
        gaming_hours = df_clean["Hours Gaming"].replace(0, 0.1)
        df_clean["StudyGamingRatio"] = df_clean["Total Study Hours"] / gaming_hours
        
        df_clean["SleepStudyRatio"] = df_clean["Sleep Hours"] / (df_clean["Total Study Hours"] + 1)
        df_clean["StudyPerUnit"] = df_clean["Total Study Hours"] / (df_clean["Academic unit"] + 0.1)
        
        return df_clean
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def train_model(df_clean):
    """Train the machine learning model"""
    try:
        # Define engineered features
        engineered_feature_names = ['Total Study Hours', 'StudyEfficiency', 'AcademicEngagement',
                                  'StressBalance', 'TimeBurden', 'StudyGamingRatio',
                                  'SleepStudyRatio', 'StudyPerUnit']
        
        # Filter only existing engineered features
        existing_engineered_features = [feat for feat in engineered_feature_names if feat in df_clean.columns]
        
        # Create X and y
        X = df_clean[existing_engineered_features]
        y = df_clean['At-Risk/Not At-Risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        
        # Train Logistic Regression model
        lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
        lr.fit(X_train_res, y_train_res)
        
        return lr, scaler, X_train_res, y_train_res, X_test_scaled, y_test, existing_engineered_features
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None, None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎓 Student At-Risk Prediction System</h1>', unsafe_allow_html=True)
    
    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df_clean = load_and_preprocess_data()
        if df_clean is not None:
            model, scaler, X_train, y_train, X_test, y_test, feature_names = train_model(df_clean)
    
    if df_clean is None or model is None:
        st.error("Failed to initialize the prediction system. Please check your data file.")
        return
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", ["Prediction", "Data Overview", "Model Performance"])
    
    if app_mode == "Prediction":
        show_prediction_interface(model, scaler, feature_names)
    elif app_mode == "Data Overview":
        show_data_overview(df_clean)
    elif app_mode == "Model Performance":
        show_model_performance(model, X_test, y_test)

def show_prediction_interface(model, scaler, feature_names):
    """Show the prediction interface"""
    st.header("Student Information Input")
    
    with st.form("student_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Study Information")
            study_weekdays = st.number_input("Study Hours (Weekdays)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
            study_weekends = st.number_input("Study Hours (Weekends)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
            academic_units = st.number_input("Academic Units", min_value=0, max_value=30, value=18, step=1)
            late_submissions = st.selectbox("Late Submissions Frequency", 
                                         options=[("Never", 1), ("Rarely", 2), ("Sometimes", 3), ("Often", 4)],
                                         format_func=lambda x: x[0])[1]
        
        with col2:
            st.subheader("Well-being & Activities")
            stress_level = st.slider("Stress Level", min_value=1, max_value=5, value=3)
            social_support = st.slider("Social Support", min_value=1, max_value=5, value=3)
            sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
            financial_difficulty = st.slider("Financial Difficulty", min_value=1, max_value=5, value=3)
        
        st.subheader("Outside Activities")
        col3, col4 = st.columns(2)
        
        with col3:
            extracurricular_involved = st.radio("Extracurricular Activities", ["No", "Yes"])
            if extracurricular_involved == "Yes":
                extracurricular_hours = st.number_input("Extracurricular Hours per Week", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
            else:
                extracurricular_hours = 0.0
            
            gaming = st.radio("Do you play games?", ["No", "Yes"])
            if gaming == "Yes":
                gaming_hours = st.number_input("Gaming Hours per Day", min_value=0.0, max_value=24.0, value=1.0, step=0.5)
            else:
                gaming_hours = 0.0
        
        with col4:
            part_time_work = st.radio("Part-time Job", ["No", "Yes"])
            if part_time_work == "Yes":
                work_hours = st.number_input("Work Hours per Week", min_value=0.0, max_value=40.0, value=10.0, step=0.5)
            else:
                work_hours = 0.0
        
        # Submit button
        submitted = st.form_submit_button("Predict Risk Level")
    
    if submitted:
        # Calculate engineered features
        user_data = {}
        user_data['Total Study Hours'] = study_weekdays + study_weekends
        user_data['StudyEfficiency'] = user_data['Total Study Hours'] / (late_submissions + 0.1)
        user_data['AcademicEngagement'] = extracurricular_hours + social_support
        user_data['StressBalance'] = stress_level - social_support
        user_data['TimeBurden'] = work_hours + gaming_hours
        
        gaming_hours_adj = gaming_hours if gaming_hours > 0 else 0.1
        user_data['StudyGamingRatio'] = user_data['Total Study Hours'] / gaming_hours_adj
        
        user_data['SleepStudyRatio'] = sleep_hours / (user_data['Total Study Hours'] + 1)
        user_data['StudyPerUnit'] = user_data['Total Study Hours'] / (academic_units + 0.1)
        
        # Create DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Ensure all feature columns are present
        for feature in feature_names:
            if feature not in user_df.columns:
                user_df[feature] = 0
        
        # Reorder columns
        user_df = user_df[feature_names]
        
        # Scale features and predict
        user_scaled = scaler.transform(user_df)
        prediction = model.predict(user_scaled)[0]
        probabilities = model.predict_proba(user_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.header("Prediction Results")
        
        if prediction == 1:
            risk_class = "at-risk"
            risk_text = "AT-RISK"
            confidence = probabilities[1]
        else:
            risk_class = "not-at-risk"
            risk_text = "NOT AT-RISK"
            confidence = probabilities[0]
        
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2>Prediction: {risk_text}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show detailed probabilities
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Probability of At-Risk", f"{probabilities[1]:.1%}")
        with col6:
            st.metric("Probability of Not At-Risk", f"{probabilities[0]:.1%}")
        
        # Show feature values
        st.subheader("Calculated Feature Values")
        feature_cols = st.columns(3)
        
        features_to_show = [
            ('Total Study Hours', user_data['Total Study Hours']),
            ('Study Efficiency', user_data['StudyEfficiency']),
            ('Academic Engagement', user_data['AcademicEngagement']),
            ('Stress Balance', user_data['StressBalance']),
            ('Time Burden', user_data['TimeBurden']),
            ('Study-Gaming Ratio', user_data['StudyGamingRatio'])
        ]
        
        for i, (name, value) in enumerate(features_to_show):
            with feature_cols[i % 3]:
                st.markdown(f'<div class="feature-value"><strong>{name}:</strong> {value:.2f}</div>', unsafe_allow_html=True)

def show_data_overview(df_clean):
    """Show data overview and statistics"""
    st.header("Dataset Overview")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Students", df_clean.shape[0])
    with col2:
        st.metric("Number of Features", df_clean.shape[1])
    with col3:
        at_risk_count = df_clean['At-Risk/Not At-Risk'].sum()
        st.metric("At-Risk Students", f"{at_risk_count} ({at_risk_count/df_clean.shape[0]*100:.1f}%)")
    
    # Show data sample
    st.subheader("Data Sample")
    st.dataframe(df_clean.head(10))
    
    # Feature distributions
    st.subheader("Feature Distributions")
    feature_to_plot = st.selectbox("Select feature to visualize", 
                                 ['Total Study Hours', 'Stress Level', 'Sleep Hours', 'StudyEfficiency'])
    
    if feature_to_plot in df_clean.columns:
        fig, ax = plt.subplots()
        df_clean[feature_to_plot].hist(bins=30, ax=ax)
        ax.set_title(f'Distribution of {feature_to_plot}')
        ax.set_xlabel(feature_to_plot)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

def show_model_performance(model, X_test, y_test):
    """Show model performance metrics"""
    st.header("Model Performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.3f}")
    with col2:
        st.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    with col3:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Import required libraries at the top
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    main()