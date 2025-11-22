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
    
    # Academic Performance Rules
    if student_data['gwa'] >= 2.25:
        risk_factors.append(f"GWA ≥ 2.25 (Current: {student_data['gwa']})")
    
    if student_data['failed_subjects'] >= 1:
        risk_factors.append(f"Failed ≥ 1 subject (Current: {student_data['failed_subjects']})")
    
    if student_data['attendance_rate'] <= 10:
        risk_factors.append(f"Attendance ≤ 10% (Current: {student_data['attendance_rate']}%)")
    
    # Study Habits Rules
    if student_data['study_weekdays'] == 0:
        risk_factors.append("Study hours (weekdays) = 0")
    
    if student_data['study_weekends'] == 0:
        risk_factors.append("Study hours (weekends) = 0")
    
    if student_data['late_submissions'] in ["sometimes", "often"]:
        risk_factors.append(f"Late submissions: {student_data['late_submissions']}")
    
    # Personal Factors Rules
    if student_data['part_time_work'] == "Yes":
        risk_factors.append("Part-time work: Yes")
    
    if student_data['work_hours'] >= 15:
        risk_factors.append(f"Work hours ≥ 15 (Current: {student_data['work_hours']})")
    
    if student_data['sleep_hours'] <= 4:
        risk_factors.append(f"Sleep ≤ 4 hours (Current: {student_data['sleep_hours']})")
    
    if student_data['gaming_hours'] >= 2:
        risk_factors.append(f"Gaming ≥ 2 hours (Current: {student_data['gaming_hours']})")
    
    # Extracurricular Rules
    if student_data['extracurricular_involvement'] == "regularly":
        risk_factors.append("Extracurricular: regularly")
    
    if student_data['extracurricular_hours'] >= 6:
        risk_factors.append(f"Extracurricular hours ≥ 6 (Current: {student_data['extracurricular_hours']})")
    
    # Well-being Rules
    if student_data['stress_level'] >= 3:
        risk_factors.append(f"Stress level ≥ 3 (Current: {student_data['stress_level']})")
    
    if student_data['financial_difficulty'] >= 3:
        risk_factors.append(f"Financial difficulty ≥ 3 (Current: {student_data['financial_difficulty']})")
    
    if student_data['social_support'] <= 3:  # Note: This is LOW social support
        risk_factors.append(f"Social support ≤ 3 (Current: {student_data['social_support']})")
    
    # Determine overall risk
    is_at_risk = len(risk_factors) > 0
    risk_score = len(risk_factors)
    
    return risk_factors, is_at_risk, risk_score

def main():
    st.subheader("Enter the student's information:")
    
    # --- ACADEMIC PERFORMANCE ---
    st.markdown("### 📊 Academic Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gwa = st.number_input(
            "General Weighted Average (GWA):",
            min_value=1.0,
            max_value=5.0,
            value=2.0,
            step=0.01,
            help="1.0 = Excellent, 5.0 = Poor"
        )
        
        failed_subjects = st.number_input(
            "Number of failed subjects:",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )
    
    with col2:
        attendance_rate = st.slider(
            "Attendance rate (%):",
            min_value=0,
            max_value=100,
            value=85,
            step=5
        )
    
    # --- STUDY HABITS ---
    st.markdown("### 📚 Study Habits")
    
    col3, col4 = st.columns(2)
    
    with col3:
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
    
    with col4:
        late_submissions = st.selectbox(
            "Late Submissions frequency:",
            options=["never", "rarely", "sometimes", "often"],
            index=0
        )
    
    # --- OUTSIDE ACTIVITIES ---
    st.markdown("### 🎯 Outside Activities")
    
    col5, col6 = st.columns(2)
    
    with col5:
        extracurricular_involvement = st.radio(
            "Extracurricular involvement:",
            ["none", "occasionally", "regularly"],
            horizontal=True
        )
        
        if extracurricular_involvement != "none":
            extracurricular_hours = st.number_input(
                "Hours spent on extracurricular activities per week:",
                min_value=0.0,
                max_value=40.0,
                value=5.0,
                step=0.5
            )
        else:
            extracurricular_hours = 0
    
    with col6:
        gaming_hours = st.number_input(
            "Hours spent playing games per day:",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.5
        )
    
    # --- PART-TIME WORK ---
    st.markdown("### 💼 Part-time Work")
    
    col7, col8 = st.columns(2)
    
    with col7:
        part_time_work = st.radio(
            "Do you work part-time?",
            ["No", "Yes"],
            horizontal=True
        )
    
    with col8:
        if part_time_work == "Yes":
            work_hours = st.number_input(
                "Work Hours per week:",
                min_value=0.0,
                max_value=40.0,
                value=10.0,
                step=0.5
            )
        else:
            work_hours = 0
    
    # --- WELL-BEING ---
    st.markdown("### 😊 Well-being")
    
    col9, col10 = st.columns(2)
    
    with col9:
        stress_level = st.slider(
            "Stress Level (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Low stress, 5 = High stress"
        )
        
        social_support = st.slider(
            "Level of Social Support (1-5 scale):",
            min_value=1,
            max_value=5,
            value=3,
            help="1 = Low support, 5 = High support"
        )
    
    with col10:
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
            value=3,
            help="1 = No difficulty, 5 = High difficulty"
        )
    
    # Predict button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Collect all student data
        student_data = {
            'gwa': gwa,
            'failed_subjects': failed_subjects,
            'attendance_rate': attendance_rate,
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
        col11, col12 = st.columns(2)
        
        with col11:
            if is_at_risk:
                st.error(f"### 🚨 Prediction: **AT-RISK**")
                st.metric("Risk Score", f"{risk_score} risk factors")
                st.warning("⚠️ This student meets one or more risk criteria that require attention.")
            else:
                st.success(f"### ✅ Prediction: **NOT AT-RISK**")
                st.metric("Risk Score", "0 risk factors")
                st.info("💡 Student does not meet any risk criteria.")
        
        with col12:
            st.subheader("Risk Assessment")
            st.write(f"**Total Risk Factors:** {risk_score}")
            st.write(f"**Risk Threshold:** ≥ 1 factor")
            st.write(f"**Status:** {'AT-RISK' if is_at_risk else 'NOT AT-RISK'}")
        
        # Display risk factors
        if risk_factors:
            st.markdown("### 🔍 Identified Risk Factors")
            
            # Group risk factors by category
            academic_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['gwa', 'failed', 'attendance', 'study', 'late'])]
            personal_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['work', 'sleep', 'gaming'])]
            activity_risks = [f for f in risk_factors if 'extracurricular' in f.lower()]
            wellbeing_risks = [f for f in risk_factors if any(keyword in f.lower() for keyword in ['stress', 'financial', 'social'])]
            
            col13, col14 = st.columns(2)
            
            with col13:
                if academic_risks:
                    st.markdown("**📚 Academic Risks:**")
                    for risk in academic_risks:
                        st.write(f"• {risk}")
                
                if personal_risks:
                    st.markdown("**👤 Personal Risks:**")
                    for risk in personal_risks:
                        st.write(f"• {risk}")
            
            with col14:
                if activity_risks:
                    st.markdown("**🎯 Activity Risks:**")
                    for risk in activity_risks:
                        st.write(f"• {risk}")
                
                if wellbeing_risks:
                    st.markdown("**😊 Well-being Risks:**")
                    for risk in wellbeing_risks:
                        st.write(f"• {risk}")
        else:
            st.success("### ✅ No Risk Factors Identified")
            st.write("This student does not meet any of the defined risk criteria.")
        
        # Summary statistics
        st.markdown("### 📈 Student Profile Summary")
        
        col15, col16, col17, col18 = st.columns(4)
        
        with col15:
            st.metric("Study Hours/Week", f"{(study_weekdays * 5) + (study_weekends * 2):.1f}")
        
        with col16:
            st.metric("Total Commitments", f"{work_hours + extracurricular_hours + (gaming_hours * 7):.1f}h/week")
        
        with col17:
            st.metric("Academic Status", f"GWA: {gwa}")
        
        with col18:
            st.metric("Well-being", f"Stress: {stress_level}/5")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ Risk Criteria")
    st.markdown("""
    **At-Risk if ANY of these apply:**
    
    **📊 Academic:**
    - GWA ≥ 2.25
    - Failed ≥ 1 subject  
    - Attendance ≤ 10%
    
    **📚 Study Habits:**
    - Study hours (weekdays) = 0
    - Study hours (weekends) = 0
    - Late submissions: sometimes/often
    
    **💼 Work & Activities:**
    - Part-time work: Yes
    - Work hours ≥ 15
    - Gaming ≥ 2 hours/day
    - Extracurricular: regularly
    - Extracurricular hours ≥ 6
    
    **😊 Well-being:**
    - Sleep ≤ 4 hours/night
    - Stress level ≥ 3
    - Financial difficulty ≥ 3
    - Social support ≤ 3
    """)
    
    st.markdown("---")
    st.markdown("**Note:** This system uses explicit rule-based criteria instead of machine learning for transparent and accurate risk assessment.")

if __name__ == "__main__":
    main()
