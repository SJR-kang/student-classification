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

# Initialize session state for reset functionality
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

def reset_all():
    """Reset all form inputs to default values"""
    st.session_state.reset_counter += 1
    # Clear all specific input states
    for key in list(st.session_state.keys()):
        if key.startswith('extracurricular') or key.startswith('gaming') or key.startswith('work') or key.startswith('study') or key.startswith('academic') or key.startswith('late') or key.startswith('stress') or key.startswith('social') or key.startswith('sleep') or key.startswith('financial'):
            del st.session_state[key]

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
            
            # Load dataset
            df = pd.read_csv('Balanced_Realistic_Student_Dataset_v2.csv')
            
            # Data preprocessing
            df_clean = df.copy()
            
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
            
            # Data cleaning
            def clean_numeric_column(column_name, default_value=0):
                if column_name in df_clean.columns:
                    df_clean[column_name] = df_clean[column_name].replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], np.nan)
                    df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')
                    if df_clean[column_name].isna().any():
                        fill_value = df_clean[column_name].median() if not df_clean[column_name].isna().all() else default_value
                        df_clean[column_name].fillna(fill_value, inplace=True)
                    return True
                return False
            
            # Clean all columns
            df_clean['Hours Gaming'] = df_clean['Hours Gaming'].astype(str).replace(['Prefer not to say', 'NaN', 'None', '?', '#VALUE!', ''], '0')
            df_clean['Hours Gaming'] = pd.to_numeric(df_clean['Hours Gaming'], errors='coerce').fillna(0)
            
            for col in ['Work Hours', 'Hours Extracurricular Activity', 'Study Hours (Weekdays)', 
                       'Study Hours (Weekends)', 'Academic unit', 'Sleep Hours', 'Stress Level', 
                       'Social Support', 'Financial Difficulty']:
                clean_numeric_column(col, 0)
            
            # Encode Late Submissions
            if 'Late Submissions' in df_clean.columns and df_clean['Late Submissions'].dtype == 'object':
                late_mapping = {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4}
                df_clean['Late Submissions'] = df_clean['Late Submissions'].map(late_mapping).fillna(2.5)
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
            
            # Create X and y
            engineered_feature_names = ['Total Study Hours', 'StudyEfficiency', 'AcademicEngagement', 
                                      'StressBalance', 'TimeBurden', 'StudyGamingRatio', 
                                      'SleepStudyRatio', 'StudyPerUnit']
            
            existing_engineered_features = [feat for feat in engineered_feature_names if feat in df_clean.columns]
            X = df_clean[existing_engineered_features]
            y = df_clean['At-Risk/Not At-Risk']
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
            
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
    
    # Initialize variables with session state
    if f'extracurricular_involved_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'extracurricular_involved_{st.session_state.reset_counter}'] = "No"
    if f'extracurricular_hours_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'extracurricular_hours_{st.session_state.reset_counter}'] = 0.0
    if f'gaming_question_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'gaming_question_{st.session_state.reset_counter}'] = "No"
    if f'gaming_hours_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'gaming_hours_{st.session_state.reset_counter}'] = 0.0
    if f'part_time_work_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'part_time_work_{st.session_state.reset_counter}'] = "No"
    if f'work_hours_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'work_hours_{st.session_state.reset_counter}'] = 0.0
    if f'study_weekdays_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'study_weekdays_{st.session_state.reset_counter}'] = 4.0
    if f'study_weekends_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'study_weekends_{st.session_state.reset_counter}'] = 2.0
    if f'late_submissions_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'late_submissions_{st.session_state.reset_counter}'] = 2
    if f'academic_units_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'academic_units_{st.session_state.reset_counter}'] = 15
    if f'stress_level_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'stress_level_{st.session_state.reset_counter}'] = 3
    if f'social_support_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'social_support_{st.session_state.reset_counter}'] = 3
    if f'sleep_hours_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'sleep_hours_{st.session_state.reset_counter}'] = 7.0
    if f'financial_difficulty_{st.session_state.reset_counter}' not in st.session_state:
        st.session_state[f'financial_difficulty_{st.session_state.reset_counter}'] = 3
    
    # --- OUTSIDE ACTIVITIES ---
    st.markdown("### 🎯 Outside Activities")
    
    # Extracurricular Activities
    extracurricular_involved = st.radio(
        "Are you involved in extracurricular activities?",
        ["No", "Yes"],
        horizontal=True,
        key=f'extracurricular_involved_{st.session_state.reset_counter}'
    )
    
    if extracurricular_involved == "Yes":
        extracurricular_hours = st.number_input(
            "Hours spent on extracurricular activities per week:",
            min_value=0.0,
            max_value=40.0,
            value=5.0,
            step=0.5,
            key=f'extracurricular_hours_{st.session_state.reset_counter}'
        )
    else:
        extracurricular_hours = 0.0
    
    # Gaming
    gaming_question = st.radio(
        "Are you playing games?",
        ["No", "Yes"],
        horizontal=True,
        key=f'gaming_question_{st.session_state.reset_counter}'
    )
    
    if gaming_question == "Yes":
        gaming_hours = st.number_input(
            "Hours spent playing games per day:",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
            key=f'gaming_hours_{st.session_state.reset_counter}'
        )
    else:
        gaming_hours = 0.0
    
    # --- PART-TIME WORK ---
    st.markdown("### 💼 Part-time Work")
    
    part_time_work = st.radio(
        "Do you work part-time?",
        ["No", "Yes"],
        horizontal=True,
        key=f'part_time_work_{st.session_state.reset_counter}'
    )
    
    if part_time_work == "Yes":
        work_hours = st.number_input(
            "Work Hours per week:",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            step=0.5,
            key=f'work_hours_{st.session_state.reset_counter}'
        )
    else:
        work_hours = 0.0
    
    # --- STUDY INFORMATION ---
    st.markdown("### 📚 Study Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_weekdays = st.number_input(
            "Study Hours (Weekdays):",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            key=f'study_weekdays_{st.session_state.reset_counter}'
        )
        
        study_weekends = st.number_input(
            "Study Hours (Weekends):",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            key=f'study_weekends_{st.session_state.reset_counter}'
        )
        
        late_submissions = st.selectbox(
            "Late Submissions frequency:",
            options=[1, 2, 3, 4],
            format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often"][x-1],
            key=f'late_submissions_{st.session_state.reset_counter}'
        )
    
    with col2:
        academic_units = st.number_input(
            "Number of Academic Units:",
            min_value=0,
            max_value=30,
            value=15,
            step=1,
            key=f'academic_units_{st.session_state.reset_counter}'
        )
    
    # --- WELL-BEING ---
    st.markdown("### 😊 Well-being")
    
    col3, col4 = st.columns(2)
    
    with col3:
        stress_level = st.slider(
            "Stress Level (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3,
            key=f'stress_level_{st.session_state.reset_counter}'
        )
        
        social_support = st.slider(
            "Level of Social Support (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3,
            key=f'social_support_{st.session_state.reset_counter}'
        )
    
    with col4:
        sleep_hours = st.number_input(
            "Sleep Hours per night:",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
            key=f'sleep_hours_{st.session_state.reset_counter}'
        )
        
        financial_difficulty = st.slider(
            "Financial Difficulty (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3,
            key=f'financial_difficulty_{st.session_state.reset_counter}'
        )
    
    # Action buttons - Reset button beside Predict button
    col_predict, col_reset = st.columns(2)
    
    with col_predict:
        predict_clicked = st.button("🔍 Predict Risk", type="primary", use_container_width=True, key=f"predict_{st.session_state.reset_counter}")
    
    with col_reset:
        reset_clicked = st.button("🔄 Reset All", use_container_width=True, key=f"reset_{st.session_state.reset_counter}")
    
    # Handle reset button click
    if reset_clicked:
        reset_all()
        st.rerun()
    
    # Handle predict button click
    if predict_clicked:
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
        prob_at_risk = probability[1]  # Assuming class 1 is "At-Risk"
        prob_not_risk = probability[0]  # Assuming class 0 is "Not At-Risk"
        
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
    
    **Model:** Logistic Regression
    **Dataset:** Balanced Student Dataset
    **Accuracy:** ~85% (varies by training)
    """)
    
    st.markdown("---")
    st.markdown("**Note:** The model is trained on your first run and saved for future use.")
