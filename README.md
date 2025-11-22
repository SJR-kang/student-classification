# Student Risk Prediction System

A Streamlit web application that predicts whether a student is at academic risk using Logistic Regression with engineered features.

## Features

- **Logistic Regression Model**: Trained on balanced student data
- **8 Engineered Features**: Calculated from student inputs
- **SMOTE Balancing**: Handles class imbalance
- **Real-time Prediction**: Instant risk assessment
- **Clean UI**: Organized input sections with clear layout

## Quick Start

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Deploy with main file: `app.py`

### Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
