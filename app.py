import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student At-Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .at-risk {
        background-color: #ffcccc;
        border: 2px solid #ff4444;
    }
    .not-at-risk {
        background-color: #ccffcc;
        border: 2px solid #44ff44;
    }
    .feature-importance {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

class StudentRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.engineered_features = [
            'Total Study Hours', 'StudyEfficiency', 'AcademicEngagement',
            'StressBalance', 'TimeBurden', 'StudyGamingRatio',
            'SleepStudyRatio', 'StudyPerUnit'
        ]
    
    def load_data(self):
        """Load and preprocess the student dataset"""
        try:
            df = pd.read_csv('Balanced_Realistic_Student_Dataset_v2.csv')
            return df
        except FileNotFoundError:
            st.error("Dataset file not found. Please make sure 'Balanced_Realistic_Student_Dataset_v2.csv' is in the same directory.")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data and create engineered features"""
        df_clean = df.copy()
        
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
        numeric_columns = ['Work Hours', 'Hours Extracurricular Activity', 'Study Hours (Weekdays)',
                          'Study Hours (Weekends)', 'Academic unit', 'Sleep Hours', 'Stress Level',
                          'Social Support', 'Financial Difficulty', 'Hours Gaming']
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].isna().any():
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
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
    
    def train_model(self, df_clean):
        """Train the machine learning model"""
        # Prepare features and target
        X = df_clean[self.engineered_features]
        y = df_clean['At-Risk/Not At-Risk']
        
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, X_test_scaled, y_test, y_pred
    
    def predict_risk(self, input_features):
        """Predict risk for new student data"""
        if self.model is None or self.scaler is None:
            st.error("Model not trained yet. Please train the model first.")
            return None
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Scale features
        input_scaled = self.scaler.transform(input_df)
        
        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probability = self.model.predict_proba(input_scaled)[0]
        
        return prediction, probability

def main():
    # Header
    st.markdown('<div class="main-header">🎓 Student At-Risk Prediction System</div>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = StudentRiskPredictor()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["🏠 Home", "📊 Data Overview", "🤖 Model Training", "🎯 Prediction", "📈 Analytics"])
    
    # Load data
    df = predictor.load_data()
    
    if df is None:
        return
    
    if app_mode == "🏠 Home":
        st.header("Welcome to the Student At-Risk Prediction System")
        st.write("""
        This application uses machine learning to identify college students who may be at risk 
        based on their academic and behavioral patterns.
        
        ### Features:
        - **Data Analysis**: Explore the student dataset and understand feature distributions
        - **Model Training**: Train Random Forest and Logistic Regression models
        - **Risk Prediction**: Predict individual student risk levels
        - **Analytics**: View model performance and feature importance
        
        ### How to use:
        1. Start with **Data Overview** to understand the dataset
        2. Train the model in **Model Training**
        3. Make predictions in **Prediction** tab
        4. Analyze results in **Analytics** tab
        """)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            at_risk_count = df['At-Risk/Not At-Risk'].sum()
            st.metric("At-Risk Students", at_risk_count)
        with col3:
            not_at_risk = len(df) - at_risk_count
            st.metric("Not At-Risk Students", not_at_risk)
        with col4:
            st.metric("Balance Ratio", f"{(at_risk_count/len(df)*100):.1f}%")
    
    elif app_mode == "📊 Data Overview":
        st.header("Dataset Overview")
        
        # Show basic info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(f"**Shape:** {df.shape}")
            st.write(f"**Columns:** {len(df.columns)}")
            
            # Target distribution
            st.subheader("Target Distribution")
            target_counts = df['At-Risk/Not At-Risk'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(target_counts.values, labels=['Not At-Risk', 'At-Risk'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.subheader("Sample Data")
            st.dataframe(df.head(10))
        
        # Show missing values
        st.subheader("Missing Values")
        missing_data = df.isnull().sum()
        st.bar_chart(missing_data)
    
    elif app_mode == "🤖 Model Training":
        st.header("Model Training")
        
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few seconds."):
                # Preprocess data
                df_clean = predictor.preprocess_data(df)
                
                # Train model
                accuracy, X_test, y_test, y_pred = predictor.train_model(df_clean)
                
                # Store in session state
                st.session_state.model_trained = True
                st.session_state.accuracy = accuracy
                st.session_state.df_clean = df_clean
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")
            
            # Show confusion matrix
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            with col2:
                st.text(classification_report(y_test, y_pred))
        
        elif 'model_trained' in st.session_state:
            st.info(f"Model already trained with accuracy: {st.session_state.accuracy:.2f}")
    
    elif app_mode == "🎯 Prediction":
        st.header("Student Risk Prediction")
        
        if 'model_trained' not in st.session_state:
            st.warning("Please train the model first in the 'Model Training' tab.")
            return
        
        st.write("Enter student information to predict risk level:")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            study_weekdays = st.slider("Study Hours (Weekdays)", 0.0, 10.0, 3.0, 0.5)
            study_weekends = st.slider("Study Hours (Weekends)", 0.0, 10.0, 2.0, 0.5)
            sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0, 0.5)
            gaming_hours = st.slider("Gaming Hours per Day", 0.0, 8.0, 1.0, 0.5)
            work_hours = st.slider("Work Hours per Week", 0, 40, 0, 5)
        
        with col2:
            academic_units = st.slider("Academic Units", 0, 24, 15, 1)
            stress_level = st.slider("Stress Level (1-5)", 1, 5, 3)
            social_support = st.slider("Social Support (1-5)", 1, 5, 3)
            financial_difficulty = st.slider("Financial Difficulty (1-5)", 1, 5, 3)
            extracurricular_hours = st.slider("Extracurricular Hours", 0, 20, 2, 1)
            late_submissions = st.selectbox("Late Submissions Frequency", 
                                          ["never", "rarely", "sometimes", "often"])
        
        # Map late submissions to numeric
        late_mapping = {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4}
        late_submissions_numeric = late_mapping[late_submissions]
        
        if st.button("Predict Risk Level"):
            # Prepare input features
            input_features = {
                'Study Hours (Weekdays)': study_weekdays,
                'Study Hours (Weekends)': study_weekends,
                'Sleep Hours': sleep_hours,
                'Hours Gaming': gaming_hours,
                'Work Hours': work_hours,
                'Academic unit': academic_units,
                'Stress Level': stress_level,
                'Social Support': social_support,
                'Financial Difficulty': financial_difficulty,
                'Hours Extracurricular Activity': extracurricular_hours,
                'Late Submissions': late_submissions_numeric
            }
            
            # Calculate engineered features
            total_study_hours = study_weekdays + study_weekends
            study_efficiency = total_study_hours / (late_submissions_numeric + 0.1)
            academic_engagement = extracurricular_hours + social_support
            stress_balance = stress_level - social_support
            time_burden = work_hours + gaming_hours
            study_gaming_ratio = total_study_hours / (gaming_hours + 0.1)
            sleep_study_ratio = sleep_hours / (total_study_hours + 1)
            study_per_unit = total_study_hours / (academic_units + 0.1)
            
            engineered_input = {
                'Total Study Hours': total_study_hours,
                'StudyEfficiency': study_efficiency,
                'AcademicEngagement': academic_engagement,
                'StressBalance': stress_balance,
                'TimeBurden': time_burden,
                'StudyGamingRatio': study_gaming_ratio,
                'SleepStudyRatio': sleep_study_ratio,
                'StudyPerUnit': study_per_unit
            }
            
            # Make prediction
            prediction, probability = predictor.predict_risk(engineered_input)
            
            # Display results
            st.subheader("Prediction Results")
            
            if prediction == 1:
                risk_class = "at-risk"
                risk_message = "🚨 This student is predicted to be AT RISK"
                risk_prob = probability[1] * 100
            else:
                risk_class = "not-at-risk"
                risk_message = "✅ This student is predicted to be NOT AT RISK"
                risk_prob = probability[0] * 100
            
            st.markdown(f"""
            <div class="prediction-box {risk_class}">
                <h3>{risk_message}</h3>
                <p><strong>Confidence:</strong> {risk_prob:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability Not At-Risk", f"{probability[0]*100:.1f}%")
            with col2:
                st.metric("Probability At-Risk", f"{probability[1]*100:.1f}%")
    
    elif app_mode == "📈 Analytics":
        st.header("Model Analytics")
        
        if 'model_trained' not in st.session_state:
            st.warning("Please train the model first in the 'Model Training' tab.")
            return
        
        # Feature Importance
        st.subheader("Feature Importance")
        
        if predictor.model is not None:
            feature_importance = pd.DataFrame({
                'feature': predictor.feature_names,
                'importance': predictor.model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance (Random Forest)')
            st.pyplot(fig)
        
        # Correlation Heatmap
        st.subheader("Feature Correlation Heatmap")
        
        if 'df_clean' in st.session_state:
            df_clean = st.session_state.df_clean
            correlation_data = df_clean[predictor.engineered_features + ['At-Risk/Not At-Risk']]
            correlation_matrix = correlation_data.corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)

if __name__ == "__main__":
    main()
