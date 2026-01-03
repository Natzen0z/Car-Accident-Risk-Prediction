# 🚗 Car Accident Risk Management System

A Python-based machine learning application that predicts car accident injury severity using historical crash data from Monroe County (2003-2015).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model Details](#model-details)
- [Project Structure](#project-structure)

## 🎯 Overview

This system uses machine learning to analyze car accident patterns and predict injury severity based on various factors such as collision type, time of day, weather conditions, and primary causes. It provides a web-based dashboard for interactive risk assessment.

## ✨ Features

- **Risk Prediction**: Predict injury severity (No Injury, Minor, Serious, Fatal) based on accident parameters
- **Risk Scoring**: Numerical risk score (0-100) for easy assessment
- **Interactive Dashboard**: Modern web interface with real-time predictions
- **Data Visualization**: Statistics and insights from historical accident data
- **REST API**: Programmatic access for integration with other systems

## 📊 Dataset

The system uses the Monroe County car crash dataset (2003-2015):

| Attribute | Details |
|-----------|---------|
| Records | 53,943 accidents |
| Time Period | 2003-2015 |
| Location | Monroe County |
| Features | Year, Month, Day, Hour, Collision Type, Injury Type, Primary Factor, Location |

### Injury Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| No Injury | 41,603 | 77.1% |
| Minor Injury | 11,136 | 20.6% |
| Serious Injury | 1,089 | 2.0% |
| Fatal | 115 | 0.2% |

## 🔧 How It Works

### 1. Data Processing

The system processes raw crash data through several steps:

```
Raw CSV Data → Data Cleaning → Feature Engineering → Model Training
```

- **Data Cleaning**: Handles missing values, standardizes categories
- **Feature Engineering**: Creates derived features like:
  - Time period (Morning/Afternoon/Evening/Night)
  - Rush hour indicator
  - Risk factor categorization
  - Collision severity score

### 2. Machine Learning Model

We use a **Random Forest Classifier** for multi-class classification:

```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Maximum tree depth
    class_weight='balanced' # Handle class imbalance
)
```

**Why Random Forest?**
- Handles categorical and numerical features well
- Robust to overfitting
- Provides feature importance rankings
- Works well with imbalanced datasets

### 3. Prediction Flow

```
User Input → Feature Encoding → Model Prediction → Risk Assessment
```

The system outputs:
- **Predicted Class**: Most likely injury severity
- **Probabilities**: Confidence for each severity level
- **Risk Level**: LOW, MEDIUM, HIGH, or CRITICAL
- **Risk Score**: 0-100 numerical score

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Navigate to the project directory**:
   ```bash
   cd car_risk_system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in place**:
   
   The CSV file should be in the parent directory:
   ```
   Dataset/
   ├── monroe county car crach 2003-2015.csv
   └── car_risk_system/
       ├── main.py
       └── ...
   ```

## 📖 Usage

### Starting the Application

```bash
python main.py
```

The application will:
1. Load and process the dataset
2. Train the model (or load existing model)
3. Start the web server

Access the dashboard at: **http://localhost:5000**

### Using the Dashboard

1. **Fill in accident parameters**:
   - Collision Type (2-Car, Single Car, etc.)
   - Day Type (Weekday/Weekend)
   - Time of Day
   - Risk Factor Category
   - Rush Hour indicator

2. **Click "Predict Risk"**

3. **View Results**:
   - Risk Score (0-100)
   - Risk Level (LOW/MEDIUM/HIGH/CRITICAL)
   - Probability breakdown

## 🔌 API Endpoints

### Predict Risk

```http
POST /predict
Content-Type: application/json

{
    "collision_type": "2-Car",
    "weekend": "Weekday",
    "time_period": "Afternoon",
    "hour": 14,
    "risk_factor": "Medium Risk",
    "rush_hour": false,
    "severity": 2
}
```

**Response**:
```json
{
    "success": true,
    "result": {
        "prediction": "No Injury",
        "probabilities": {
            "No Injury": 75.2,
            "Minor Injury": 20.1,
            "Serious Injury": 4.2,
            "Fatal": 0.5
        },
        "risk_level": "LOW",
        "risk_score": 7.8
    }
}
```

### Get Statistics

```http
GET /statistics
```

### Get Model Info

```http
GET /api/model-info
```

## 🤖 Model Details

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| Collision Type | Categorical | Type of collision (1-Car, 2-Car, etc.) |
| Weekend? | Categorical | Weekday or Weekend |
| Time_Period | Categorical | Morning/Afternoon/Evening/Night |
| Risk_Factor_Category | Categorical | Low/Medium/High Risk |
| Is_Rush_Hour | Binary | During rush hour (7-9AM, 4-6PM) |
| Collision_Severity | Numeric | Severity score (1-4) |
| Hour_Numeric | Numeric | Hour of day (0-23) |

### Risk Level Calculation

| Level | Condition |
|-------|-----------|
| CRITICAL | Fatal predicted OR high-risk probability > 30% |
| HIGH | Serious Injury predicted OR high-risk probability > 15% |
| MEDIUM | Minor Injury predicted |
| LOW | No Injury predicted |

### Risk Score Formula

```python
risk_score = (
    P(No Injury) × 0 +
    P(Minor Injury) × 25 +
    P(Serious Injury) × 60 +
    P(Fatal) × 100
)
```

## 📁 Project Structure

```
car_risk_system/
├── main.py              # Flask application entry point
├── data_processor.py    # Data loading and preprocessing
├── model.py             # ML model training and prediction
├── requirements.txt     # Python dependencies
├── trained_model.joblib # Saved model (auto-generated)
├── README.md            # This file
├── templates/
│   └── index.html       # Web dashboard template
└── static/
    └── style.css        # Dashboard styling
```

## 📝 License

This project is for educational purposes. The dataset is sourced from Monroe County public records.

## 🙏 Acknowledgments

- Monroe County for providing the crash data
- scikit-learn team for the machine learning library
- Flask team for the web framework

---

**Built with ❤️ using Python, Flask, and scikit-learn**
