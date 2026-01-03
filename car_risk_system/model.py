"""
Machine Learning Model Module
Handles model training, evaluation, and predictions for car accident risk assessment.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path


class RiskModel:
    """Random Forest Classifier for predicting car accident injury severity."""
    
    def __init__(self):
        """Initialize the RiskModel."""
        self.model = None
        self.label_encoders = {}
        self.target_encoder = None
        self.feature_columns = []
        self.is_trained = False
        self.model_path = Path(__file__).parent / 'trained_model.joblib'
        
    def prepare_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Encode categorical features for model input.
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the encoders (True for training, False for prediction)
            
        Returns:
            Encoded feature array
        """
        X_encoded = X.copy()
        
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    X_encoded[col] = X[col].astype(str).apply(
                        lambda x: self._safe_transform(self.label_encoders[col], x)
                    )
                else:
                    X_encoded[col] = 0
        
        return X_encoded.values
    
    def _safe_transform(self, encoder: LabelEncoder, value: str) -> int:
        """Safely transform a value, returning 0 for unseen categories."""
        try:
            return encoder.transform([value])[0]
        except ValueError:
            return 0
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_columns = X.columns.tolist()
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Prepare features
        X_encoded = self.prepare_features(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Random Forest with class weights to handle imbalance
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        target_names = self.target_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'feature_importance': feature_importance,
            'classes': target_names.tolist()
        }
        
        print(f"\nModel Training Complete!")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        
        return metrics
    
    def predict(self, input_data: dict) -> dict:
        """
        Predict injury risk for given accident parameters.
        
        Args:
            input_data: Dictionary with accident parameters
            
        Returns:
            Dictionary with prediction and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet!")
        
        # Create DataFrame from input
        X = pd.DataFrame([input_data])
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 'Unknown' if col in self.label_encoders else 0
        
        X = X[self.feature_columns]
        
        # Encode features
        X_encoded = self.prepare_features(X, fit=False)
        
        # Get prediction and probabilities
        prediction = self.model.predict(X_encoded)[0]
        probabilities = self.model.predict_proba(X_encoded)[0]
        
        # Decode prediction
        predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        prob_dict = dict(zip(self.target_encoder.classes_, probabilities))
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(predicted_class, prob_dict)
        
        return {
            'prediction': predicted_class,
            'probabilities': prob_dict,
            'risk_level': risk_level,
            'risk_score': self._calculate_risk_score(prob_dict)
        }
    
    def _calculate_risk_level(self, prediction: str, probabilities: dict) -> str:
        """Calculate overall risk level based on prediction and probabilities."""
        high_risk_prob = probabilities.get('Fatal', 0) + probabilities.get('Serious Injury', 0)
        
        if prediction == 'Fatal' or high_risk_prob > 0.3:
            return 'CRITICAL'
        elif prediction == 'Serious Injury' or high_risk_prob > 0.15:
            return 'HIGH'
        elif prediction == 'Minor Injury':
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _calculate_risk_score(self, probabilities: dict) -> float:
        """Calculate a numerical risk score from 0-100."""
        weights = {
            'No Injury': 0,
            'Minor Injury': 25,
            'Serious Injury': 60,
            'Fatal': 100
        }
        
        score = sum(probabilities.get(k, 0) * v for k, v in weights.items())
        return round(score, 1)
    
    def save_model(self):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model!")
        
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_columns': self.feature_columns
        }
        
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load a trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_path.exists():
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.label_encoders = model_data['label_encoders']
            self.target_encoder = model_data['target_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            print("Model loaded successfully!")
            return True
        return False
