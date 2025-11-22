import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title and description
st.title("🎓 Student Risk Prediction System")
st.markdown("---")

def calculate_risk_factors(student_data):
    """
    Calculate risk factors based on the specified rules
    Returns: list of risk factors and overall risk status
    """
    risk_factors = []
    
    # Study Habits Rules
    if student_data['study_weekdays'] == 0:
        risk_factors.append("❌ Study hours (weekdays) = 0")
    
    if student_data['study_weekends'] == 0:
        risk_factors.append("❌ Study hours (weekends) = 0")
    
    if student_data['late_submissions'] in ["sometimes", "often"]:
        risk_factors.append(f"❌ Late submissions: {student_data['late_submissions']}")
    
    # Personal Factors Rules
    if student_data['part_time_work'] == "Yes":
        risk_factors.append("❌ Part-time work: Yes")
    
    if student_data['work_hours'] >= 15:
        risk_factors.append(f"❌ Work hours ≥ 15 (Current: {student_data['work_hours']})")
    
    if student_data['sleep_hours'] <= 4:
        risk_factors.append(f"❌ Sleep ≤ 4 hours (Current: {student_data['sleep_hours']})")
    
    if student_data['gaming_hours'] >= 2:
        risk_factors.append(f"❌ Gaming ≥ 2 hours (Current: {student_data['gaming_hours']})")
    
    # Extracurricular Rules
    if student_data['extracurricular_involvement'] == "regularly":
        risk_factors.append("❌ Extracurricular: regularly")
    
    if student_data['extracurricular_hours'] >= 6:
        risk_factors.append(f"❌ Extracurricular hours ≥ 6 (Current: {student_data['extracurricular_hours']})")
    
    # Well-being Rules - FIXED LOGIC
    if student_data['stress_level'] >= 4:  # Only high stress (4-5) is risky
        risk_factors.append(f"❌ High stress level ≥ 4 (Current: {student_data['stress_level']})")
    
    if student_data['financial_difficulty'] >= 4:  # Only high financial difficulty (4-5) is risky
        risk_factors.append(f"❌ High financial difficulty ≥ 4 (Current: {student_data['financial_difficulty']})")
    
    # REMOVED: Social support as a risk factor - low support alone shouldn't flag as risk
    
    # Determine overall risk
    is_at_risk = len(risk_factors) > 0
    risk_score = len(risk_factors)
    
    return risk_factors, is_at_risk, risk_score

def main():
    st.subheader("Enter the student's information:")
    
    # --- STUDY HABITS ---
    st.markdown("### 📚 Study Habits")
    
    col1, col2 = st.columns(2)
    
    with col1:
        study_weekdays = st.number_input(
            "Study Hours (Weekdays):",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5,
            help="Recommended: 2-6 hours per weekday"
        )
        
        study_weekends = st.number_input(
            "Study Hours (Weekends):",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            help="Recommended: 1-4 hours per weekend day"
        )
    
    with col2:
        late_submissions = st.selectbox(
            "Late Submissions frequency:",
            options=["never", "rarely", "sometimes", "often"],
            index=0,
            help="Never/Rarely = Low risk, Sometimes/Often = High risk"
        )
    
    # --- OUTSIDE ACTIVITIES ---
    st.markdown("### 🎯 Outside Activities")
    
    col3, col4 = st.columns(2)
    
    with col3:
        extracurricular_involvement = st.radio(
            "Extracurricular involvement:",
            ["none", "occasionally", "regularly"],
            horizontal=True,
            help="Regular involvement may indicate time management issues"
        )
        
        if extracurricular_involvement != "none":
            extracurricular_hours = st.number_input(
                "Hours spent on extracurricular activities per week:",
                min_value=0.0,
                max_value=40.0,
                value=5.0,
                step=0.5,
                help="≥6 hours/week may impact studies"
            )
        else:
            extracurricular_hours = 0
    
    with col4:
        gaming_hours = st.number_input(
            "Hours spent playing games per day:",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5,
            help="≥2 hours/day may impact academic performance"
        )
    
    # --- PART-TIME WORK ---
    st.markdown("### 💼 Part-time Work")
    
    col5, col6 = st.columns(2)
    
    with col5:
        part_time_work = st.radio(
            "Do you work part-time?",
            ["No", "Yes"],
            horizontal=True,
            help="Part-time work can impact study time"
        )
    
    with col6:
        if part_time_work == "Yes":
            work_hours = st.number_input(
                "Work Hours per week:",
                min_value=0.0,
                max_value=40.0,
                value=10.0,
                step=0.5,
                help="≥15 hours/week may impact studies"
            )
        else:
            work_hours = 0
    
    # --- WELL-BEING ---
    st.markdown("### 😊 Well-being")
    
    col7, col8 = st.columns(2)
    
    with col7:
        stress_level = st.slider(
            "Stress Level (1-5 scale):",
            min_value=1,
            max_value=5,
            value=2,
            help="1-2 = Low, 3 = Moderate, 4-5 = High risk"
        )
        
        social_support = st.slider(
            "Level of Social Support (1-5 scale):",
            min_value=1,
            max_value=5,
            value=4,
            help="1-2 = Low, 3-5 = Adequate support"
        )
    
    with col8:
        sleep_hours = st.number_input(
            "Sleep Hours per night:",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
            help="≤4 hours/night indicates sleep deprivation"
        )
        
        financial_difficulty = st.slider(
            "Financial Difficulty (1-5 scale):",
            min_value=1,
            max_value=5,
            value=2,
            help="1-3 = Manageable, 4-5 = High difficulty"
        )
    
    # Predict button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Collect all student data
        student_data = {
            'study_weekdays': study_weekdays,
            'study_weekends': study_weekends,
            'late_submissions': late_submissions,
            'part_time_work': part_time_work,
            'work_hours': work_hours,
            'sleep_hours': sleep_hours,
            'gaming_hours': gaming_hours,
            'extracurricular_involvement': extracurricular_involvement,
            'extracurricular_hours': extracurricular_hours,
            'stress_level': stress_level,
            'financial_difficulty': financial_difficulty,
            'social_support': social_support
        }
        
        # Calculate risk factors
        risk_factors, is_at_risk, risk_score = calculate_risk_factors(student_data)
        
        # === RESULTS ===
        st.markdown("---")
        st.markdown("## 📊 PREDICTION RESULT")
        st.markdown("---")
        
        # Prediction and risk score
        col9, col10 = st.columns(2)
        
        with col9:
            if is_at_risk:
                st.error(f"### 🚨 Prediction: **AT-RISK**")
                st.metric("Risk Score", f"{risk_score} risk factors")
                if risk_score >= 3:
                    st.error("🚨 Multiple high-risk factors detected - immediate attention recommended")
                else:
                    st.warning("⚠️ Some risk factors detected - monitoring recommended")
            else:
                st.success(f"### ✅ Prediction: **NOT AT-RISK**")
                st.metric("Risk Score", "0 risk factors")
                st.info("💡 Student shows good academic habits and well-being")
        
        with col10:
            st.subheader("Risk Assessment")
            st.write(f"**Total Risk Factors:** {risk_score}")
            st.write(f"**Risk Threshold:** ≥ 1 factor")
            st.write(f"**Status:** {'AT-RISK' if is_at_risk else 'NOT AT-RISK'}")
            
            # Positive factors
            positive_factors = []
            if study_weekdays >= 2:
                positive_factors.append("✅ Good weekday study habits")
            if study_weekends >= 1:
                positive_factors.append("✅ Good weekend study habits")
            if late_submissions in ["never", "rarely"]:
                positive_factors.append("✅ Timely submissions")
            if sleep_hours >= 6:
                positive_factors.append("✅ Adequate sleep")
            if stress_level <= 3:
                positive_factors.append("✅ Manageable stress")
            if social_support >= 3:
                positive_factors.append("✅ Good social support")
            
            if positive_factors:
                st.subheader("✅ Positive Factors")
                for factor in positive_factors:
                    st.write(factor)
        
        # Display risk factors
        if risk_factors:
            st.markdown("### 🔍 Identified Risk Factors")
            
            # Group risk factors by category
            study_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['study', 'late'])]
            personal_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['work', 'sleep', 'gaming'])]
            activity_risks = [f for f in risk_factors if 'extracurricular' in f.lower()]
            wellbeing_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['stress', 'financial'])]
            
            col11, col12 = st.columns(2)
            
            with col11:
                if study_risks:
                    st.markdown("**📚 Study Risks:**")
                    for risk in study_risks:
                        st.write(f"{risk}")
                
                if personal_risks:
                    st.markdown("**👤 Personal Risks:**")
                    for risk in personal_risks:
                        st.write(f"{risk}")
            
            with col12:
                if activity_risks:
                    st.markdown("**🎯 Activity Risks:**")
                    for risk in activity_risks:
                        st.write(f"{risk}")
                
                if wellbeing_risks:
                    st.markdown("**😊 Well-being Risks:**")
                    for risk in wellbeing_risks:
                        st.write(f"{risk}")
        else:
            st.success("### ✅ No Risk Factors Identified")
            st.write("This student shows healthy academic habits and good well-being.")
        
        # Summary statistics
        st.markdown("### 📈 Student Profile Summary")
        
        col13, col14, col15, col16 = st.columns(4)
        
        with col13:
            total_study = (study_weekdays * 5) + (study_weekends * 2)
            status = "✅ Good" if total_study >= 15 else "⚠️ Low" if total_study < 10 else "🟡 Moderate"
            st.metric("Study Hours/Week", f"{total_study:.1f}", status)
        
        with col14:
            total_commitments = work_hours + extracurricular_hours + (gaming_hours * 7)
            status = "⚠️ High" if total_commitments > 20 else "✅ Manageable"
            st.metric("Total Commitments", f"{total_commitments:.1f}h/week", status)
        
        with col15:
            status = "✅ Low" if stress_level <= 2 else "⚠️ High" if stress_level >= 4 else "🟡 Moderate"
            st.metric("Stress Level", f"{stress_level}/5", status)
        
        with col16:
            status = "❌ Low" if sleep_hours < 5 else "✅ Good" if sleep_hours >= 7 else "🟡 Adequate"
            st.metric("Sleep Quality", f"{sleep_hours}h/night", status)

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Risk Criteria")
    st.markdown("""
    **At-Risk if ANY of these apply:**
    
    **📚 Study Habits:**
    - Study hours (weekdays) = 0
    - Study hours (weekends) = 0
    - Late submissions: sometimes/often
    
    **💼 Work & Activities:**
    - Part-time work: Yes
    - Work hours ≥ 15/week
    - Gaming ≥ 2 hours/day
    - Extracurricular: regularly
    - Extracurricular hours ≥ 6/week
    
    **😊 Well-being:**
    - Sleep ≤ 4 hours/night
    - Stress level ≥ 4
    - Financial difficulty ≥ 4
    """)
    
    st.markdown("---")
    st.markdown("""
    **Healthy Ranges:**
    - Study: 15-25 hours/week
    - Sleep: 7-9 hours/night  
    - Stress: 1-3/5
    - Commitments: <20 hours/week
    """)

if __name__ == "__main__":
    main()
