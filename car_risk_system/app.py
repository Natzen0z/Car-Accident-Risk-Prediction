import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Car Accident Risk Predictor", layout="centered")

st.markdown("""
<style>
.main-title {
    font-size:2.5rem;
    font-weight:bold;
    color:#e74c3c;
    text-align:center;
    margin-bottom:0.5em;
}
.subtitle {
    color:#555;
    text-align:center;
    margin-bottom:1.5em;
}
.stButton>button {
    background-color: #e74c3c;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 0.5em 2em;
}
.stButton>button:hover {
    background-color: #c0392b;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚗 Car Accident Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict injury severity using Machine Learning</div>', unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("../monroe county car crach 2003-2015.csv", encoding='latin-1')
    return df

data = load_data()

# =========================
# DATA PREPROCESSING
# =========================
# Map Injury Type to numeric
injury_map = {
    'No injury/unknown': 0,
    'Non-incapacitating': 1,
    'Incapacitating': 2,
    'Fatal': 3
}
data['Injury_Numeric'] = data['Injury Type'].map(injury_map).fillna(0).astype(int)

# Map Collision Type
collision_map = {
    '1-Car': 1, '2-Car': 2, '3+ Cars': 3,
    'Pedestrian': 4, 'Cyclist': 5, 'Moped/Motorcycle': 6, 'Bus': 7
}
data['Collision_Numeric'] = data['Collision Type'].map(collision_map).fillna(2).astype(int)

# Weekend binary
data['Is_Weekend'] = data['Weekend?'].apply(lambda x: 1 if x == 'Weekend' else 0)

# Extract hour
def extract_hour(h):
    try:
        h = int(h)
        return h // 100 if h >= 100 else h
    except:
        return 12
data['Hour_Numeric'] = data['Hour'].apply(extract_hour)

# Time period
def get_time_period(h):
    if 5 <= h < 12: return 0  # Morning
    elif 12 <= h < 17: return 1  # Afternoon
    elif 17 <= h < 21: return 2  # Evening
    else: return 3  # Night
data['Time_Period'] = data['Hour_Numeric'].apply(get_time_period)

# Risk factor from Primary Factor
def categorize_risk(factor):
    factor = str(factor).upper()
    high_risk = ['SPEED', 'LEFT OF CENTER', 'DISREGARD', 'RAN OFF', 'ALCOHOL', 'DRUG']
    for r in high_risk:
        if r in factor:
            return 2  # High
    medium_risk = ['YIELD', 'FOLLOWING', 'IMPROPER', 'LANE']
    for r in medium_risk:
        if r in factor:
            return 1  # Medium
    return 0  # Low

data['Risk_Factor'] = data['Primary Factor'].apply(categorize_risk)

# Rush hour
data['Is_Rush_Hour'] = data['Hour_Numeric'].apply(lambda x: 1 if (7<=x<=9) or (16<=x<=18) else 0)

# Select features
features = ['Collision_Numeric', 'Is_Weekend', 'Time_Period', 'Hour_Numeric', 'Risk_Factor', 'Is_Rush_Hour']
X = data[features]
y = data['Injury_Numeric']

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("<h2 style='color:#e74c3c;text-align:center;'>🚗 Car Risk</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#888;margin-bottom:1em;'>Accident Risk Prediction</div>", unsafe_allow_html=True)
    menu = st.radio(
        "Menu",
        ["EDA Dataset",
         "Predict Risk",
         "Model Evaluation",
         "SMOTE vs NearMiss",
         "Decision Tree",
         "Random Forest",
         "About"]
    )
    st.markdown("---")
    st.markdown("<div style='font-size:0.9em;color:#555;'>Monroe County Dataset<br>2003-2015</div>", unsafe_allow_html=True)

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_sc, y_train)
y_pred = rf.predict(X_test_sc)

# =========================
# MENU 1: EDA
# =========================
if menu == "EDA Dataset":
    st.header("Exploratory Data Analysis (EDA)")
    
    st.subheader("Dataset Preview")
    st.dataframe(data[['Collision Type', 'Injury Type', 'Weekend?', 'Hour', 'Primary Factor']].head(10))
    
    st.subheader("Injury Type Distribution")
    fig, ax = plt.subplots()
    data['Injury Type'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=['lightgreen', 'yellow', 'orange', 'red'],
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)
    
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr_data = data[features + ['Injury_Numeric']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# =========================
# MENU 2: PREDICT RISK
# =========================
elif menu == "Predict Risk":
    st.header("Predict Accident Risk")
    st.markdown("<div style='color:#e74c3c;font-weight:bold;'>Enter accident details:</div>", unsafe_allow_html=True)
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            collision = st.selectbox("Collision Type", 
                options=[1, 2, 3, 4, 5, 6, 7],
                format_func=lambda x: ['1-Car', '2-Car', '3+ Cars', 'Pedestrian', 'Cyclist', 'Motorcycle', 'Bus'][x-1]
            )
            weekend = st.selectbox("Day Type", [0, 1], format_func=lambda x: "Weekday" if x==0 else "Weekend")
            time_period = st.selectbox("Time of Day", [0, 1, 2, 3], 
                format_func=lambda x: ['Morning', 'Afternoon', 'Evening', 'Night'][x]
            )
        with col2:
            hour = st.number_input("Hour (0-23)", 0, 23, 12)
            risk_factor = st.selectbox("Risk Factor", [0, 1, 2],
                format_func=lambda x: ['Low Risk', 'Medium Risk', 'High Risk'][x]
            )
            rush_hour = st.selectbox("Rush Hour?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        
        submit = st.form_submit_button("Predict")
    
    if submit:
        input_data = pd.DataFrame({
            'Collision_Numeric': [collision],
            'Is_Weekend': [weekend],
            'Time_Period': [time_period],
            'Hour_Numeric': [hour],
            'Risk_Factor': [risk_factor],
            'Is_Rush_Hour': [rush_hour]
        })
        input_sc = scaler.transform(input_data)
        pred = rf.predict(input_sc)[0]
        prob = rf.predict_proba(input_sc)[0]
        
        injury_names = ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatal']
        risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
        if pred == 3:
            st.error(f"⚠️ RISK LEVEL: {risk_levels[pred]}")
        elif pred == 2:
            st.warning(f"⚠️ RISK LEVEL: {risk_levels[pred]}")
        elif pred == 1:
            st.info(f"⚠️ RISK LEVEL: {risk_levels[pred]}")
        else:
            st.success(f"✅ RISK LEVEL: {risk_levels[pred]}")
        
        st.markdown(f"**Predicted Outcome:** {injury_names[pred]}")
        
        st.subheader("Probabilities")
        prob_df = pd.DataFrame({
            'Outcome': injury_names,
            'Probability': [f"{p*100:.1f}%" for p in prob]
        })
        st.dataframe(prob_df)

# =========================
# MENU 3: MODEL EVALUATION
# =========================
elif menu == "Model Evaluation":
    st.header("Model Evaluation")
    
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {accuracy:.2%}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Injury', 'Minor', 'Serious', 'Fatal'],
                    yticklabels=['No Injury', 'Minor', 'Serious', 'Fatal'],
                    ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, 
            target_names=['No Injury', 'Minor', 'Serious', 'Fatal'],
            output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    st.subheader("Feature Importance")
    feat_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feat_df, ax=ax, palette='Reds')
    st.pyplot(fig)

# =========================
# MENU 4: SMOTE vs NearMiss
# =========================
elif menu == "SMOTE vs NearMiss":
    st.header("SMOTE vs NearMiss Comparison")
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train_sc, y_train)
    rf_smote = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_smote.fit(X_smote, y_smote)
    y_pred_smote = rf_smote.predict(X_test_sc)
    acc_smote = accuracy_score(y_test, y_pred_smote)
    
    # NearMiss
    nm = NearMiss()
    X_nm, y_nm = nm.fit_resample(X_train_sc, y_train)
    rf_nm = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_nm.fit(X_nm, y_nm)
    y_pred_nm = rf_nm.predict(X_test_sc)
    acc_nm = accuracy_score(y_test, y_pred_nm)
    
    st.subheader("Accuracy Comparison")
    acc_df = pd.DataFrame({
        'Method': ['SMOTE', 'NearMiss'],
        'Accuracy': [acc_smote, acc_nm]
    })
    st.dataframe(acc_df)
    
    fig, ax = plt.subplots()
    sns.barplot(x='Method', y='Accuracy', data=acc_df, ax=ax, palette=['green', 'orange'])
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    st.subheader("Conclusion")
    if acc_smote > acc_nm:
        st.success(f"SMOTE is recommended with higher accuracy ({acc_smote:.2%}) vs NearMiss ({acc_nm:.2%})")
    else:
        st.success(f"NearMiss is recommended with higher accuracy ({acc_nm:.2%}) vs SMOTE ({acc_smote:.2%})")

# =========================
# MENU 5: DECISION TREE
# =========================
elif menu == "Decision Tree":
    st.header("Decision Tree Model")
    
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
    dt.fit(X_train_sc, y_train)
    y_pred_dt = dt.predict(X_test_sc)
    
    st.subheader("Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_dt))
    st.write("Precision:", precision_score(y_test, y_pred_dt, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred_dt, average='weighted'))
    st.write("F1-Score:", f1_score(y_test, y_pred_dt, average='weighted'))
    
    st.subheader("Decision Tree Visualization")
    dot_data = tree.export_graphviz(
        dt, out_file=None, filled=True, rounded=True,
        feature_names=features,
        class_names=['No Injury', 'Minor', 'Serious', 'Fatal']
    )
    st.graphviz_chart(dot_data)

# =========================
# MENU 6: RANDOM FOREST
# =========================
elif menu == "Random Forest":
    st.header("Random Forest Model")
    
    st.subheader("Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
    st.write("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
    
    st.subheader("Sample Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        rf.estimators_[0],
        feature_names=features,
        class_names=['No Injury', 'Minor', 'Serious', 'Fatal'],
        filled=True, rounded=True, fontsize=8
    )
    st.pyplot(fig)

# =========================
# MENU 7: ABOUT
# =========================
elif menu == "About":
    st.header("About This App")
    st.markdown("""
    **Car Accident Risk Predictor** is a machine learning application to analyze 
    car accident data and predict injury severity.
    
    **Features:**
    - Exploratory Data Analysis
    - Risk Prediction with Random Forest
    - Model Evaluation & Comparison
    - SMOTE vs NearMiss analysis
    - Decision Tree & Random Forest visualization
    
    **Dataset:** Monroe County Car Crash 2003-2015 (53,943 records)
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<div style='text-align:center;color:#888;'><b>Streamlit ML | Car Accident Risk Dataset</b></div>", unsafe_allow_html=True)
