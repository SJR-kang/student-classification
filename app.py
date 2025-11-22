import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title and description
st.title("🎓 Student Risk Prediction System")
st.markdown("---")

# Load or train model
@st.cache_resource
def load_model_and_scaler():
    try:
        # Try to load pre-trained model
        if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            st.success("✅ Pre-trained model loaded successfully!")
            return model, scaler
        else:
            # Train model if not exists
            st.info("🔄 Training model... This may take a moment.")
            df = pd.read_csv('Balanced_Realistic_Student_Dataset_v2.csv')
            
            # Data preprocessing
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

            # Rename columns
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

            # Data cleaning function
            def clean_numeric_column(column_name, default_value=0):
                if column_name in df_clean.columns:
                    df_clean[column_name] = df_clean[column_name].replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], np.nan)
                    df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
                    if df_clean[column_name].isna().any():
                        fill_value = df_clean[column_name].median() if not df_clean[column_name].isna().all() else default_value
                        df_clean[column_name].fillna(fill_value, inplace=True)
                    return True
                return False

            # Clean specific columns
            df_clean['Hours Gaming'] = df_clean['Hours Gaming'].astype(str)
            df_clean['Hours Gaming'] = df_clean['Hours Gaming'].replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], '0')
            df_clean['Hours Gaming'] = pd.to_numeric(df_clean['Hours Gaming'], errors='coerce')
            df_clean['Hours Gaming'].fillna(0, inplace=True)

            # Clean other numeric columns
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
            
            # Define engineered features for model training
            engineered_feature_names = [
                'Total Study Hours', 'StudyEfficiency', 'AcademicEngagement',
                'StressBalance', 'TimeBurden', 'StudyGamingRatio',
                'SleepStudyRatio', 'StudyPerUnit'
            ]
            
            # Filter only existing engineered features
            existing_engineered_features = [feat for feat in engineered_feature_names if feat in df_clean.columns]
            
            # Prepare features and target
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
            
            # Save model and scaler
            with open('model.pkl', 'wb') as f:
                pickle.dump(lr, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
            st.success("✅ Model trained and saved successfully!")
            return lr, scaler
            
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None

# Load model
model, scaler = load_model_and_scaler()

if model and scaler:
    st.subheader("Enter the student's information:")
    
    # Initialize variables
    extracurricular_hours = 0
    work_hours = 0
    gaming_hours = 0
    
    # --- OUTSIDE ACTIVITIES ---
    st.markdown("### 🎯 Outside Activities")
    
    # Extracurricular Activities
    extracurricular_involved = st.radio(
        "Are you involved in extracurricular activities?",
        ["No", "Yes"],
        horizontal=True
    )
    
    if extracurricular_involved == "Yes":
        extracurricular_hours = st.number_input(
            "Hours spent on extracurricular activities per week:",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5
        )
    
    # Gaming
    gaming_question = st.radio(
        "Are you playing games?",
        ["No", "Yes"],
        horizontal=True
    )
    
    if gaming_question == "Yes":
        gaming_hours = st.number_input(
            "Hours spent playing games per day:",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5
        )
    
    # --- PART-TIME WORK ---
    st.markdown("### 💼 Part-time Work")
    
    part_time_work = st.radio(
        "Do you work part-time?",
        ["No", "Yes"],
        horizontal=True
    )
    
    if part_time_work == "Yes":
        work_hours = st.number_input(
            "Work Hours per week:",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            step=0.5
        )
    
    # --- STUDY INFORMATION ---
    st.markdown("### 📚 Study Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_weekdays = st.number_input(
            "Study Hours (Weekdays):",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )
        
        study_weekends = st.number_input(
            "Study Hours (Weekends):",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5
        )
        
        late_submissions = st.selectbox(
            "Late Submissions frequency:",
            options=[1, 2, 3, 4],
            format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often"][x-1]
        )
    
    with col2:
        academic_units = st.number_input(
            "Number of Academic Units:",
            min_value=0,
            max_value=30,
            value=15,
            step=1
        )
    
    # --- WELL-BEING ---
    st.markdown("### 😊 Well-being")
    
    col3, col4 = st.columns(2)
    
    with col3:
        stress_level = st.slider(
            "Stress Level (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3
        )
        
        social_support = st.slider(
            "Level of Social Support (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3
        )
    
    with col4:
        sleep_hours = st.number_input(
            "Sleep Hours per night:",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )
        
        financial_difficulty = st.slider(
            "Financial Difficulty (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3
        )
    
    # Predict button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Calculate engineered features
        total_study_hours = study_weekdays + study_weekends
        study_efficiency = total_study_hours / (late_submissions + 0.1)
        academic_engagement = extracurricular_hours + social_support
        stress_balance = stress_level - social_support
        time_burden = work_hours + gaming_hours
        study_gaming_ratio = total_study_hours / (gaming_hours if gaming_hours > 0 else 0.1)
        sleep_study_ratio = sleep_hours / (total_study_hours + 1)
        study_per_unit = total_study_hours / (academic_units if academic_units > 0 else 0.1)
        
        # Create feature array
        features = np.array([[
            total_study_hours, study_efficiency, academic_engagement,
            stress_balance, time_burden, study_gaming_ratio,
            sleep_study_ratio, study_per_unit
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Determine probabilities
        classes = model.classes_
        prob_at_risk = probability[1]  # Assuming 1 is At-Risk
        prob_not_risk = probability[0]  # Assuming 0 is Not At-Risk
        
        # === RESULTS ===
        st.markdown("---")
        st.markdown("## 📊 PREDICTION RESULT")
        st.markdown("---")
        
        # Prediction and confidence
        col5, col6 = st.columns(2)
        
        with col5:
            if prediction == 1:  # At-Risk
                st.error(f"### Prediction: **AT-RISK**")
                confidence = prob_at_risk
            else:  # Not At-Risk
                st.success(f"### Prediction: **NOT AT-RISK**")
                confidence = prob_not_risk
            
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col6:
            st.subheader("Probability Breakdown")
            st.write(f"**Not At-Risk:** {prob_not_risk:.1%}")
            st.write(f"**At-Risk:** {prob_at_risk:.1%}")
        
        # Key features display
        st.markdown("### 🔍 Key Calculated Features")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.write(f"**• Total Study Hours:** {total_study_hours:.1f}")
            st.write(f"**• Study Efficiency:** {study_efficiency:.1f}")
            st.write(f"**• Academic Engagement:** {academic_engagement:.1f}")
            st.write(f"**• Stress Balance:** {stress_balance:.1f}")
        
        with col8:
            st.write(f"**• Time Burden:** {time_burden:.1f}")
            st.write(f"**• Study-Gaming Ratio:** {study_gaming_ratio:.1f}")
            st.write(f"**• Financial Difficulty:** {financial_difficulty}")
            st.write(f"**• Part-time Work:** {'Yes' if part_time_work == 'Yes' else 'No'}")
            if part_time_work == "Yes":
                st.write(f"**• Work Hours per week:** {work_hours}")

else:
    st.error("❌ Model not available. Please check your dataset.")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool predicts student academic risk using Logistic Regression 
    with 8 engineered features based on study habits and personal factors.
    
    **Features used:**
    - Total Study Hours
    - Study Efficiency  
    - Academic Engagement
    - Stress Balance
    - Time Burden
    - Study-Gaming Ratio
    - Sleep-Study Ratio
    - Study per Unit
    
    **How it works:**
    1. Enter student information
    2. System calculates engineered features
    3. Machine learning model makes prediction
    4. Get risk assessment with confidence score
    """)
    
    st.markdown("---")
    st.markdown("**Dataset:** Balanced_Realistic_Student_Dataset_v2.csv")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Accuracy:** ~85%")
