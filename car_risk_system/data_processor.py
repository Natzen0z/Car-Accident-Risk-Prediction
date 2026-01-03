"""
Data Processor Module
Handles data loading, cleaning, and preprocessing for the car accident dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataProcessor:
    """Handles all data processing operations for the car accident dataset."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the DataProcessor.
        
        Args:
            data_path: Path to the CSV dataset file
        """
        self.data_path = data_path
        self.df = None
        self.feature_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the car accident dataset from CSV.
        
        Returns:
            DataFrame with the loaded data
        """
        if self.data_path is None:
            # Default path relative to this file
            base_path = Path(__file__).parent.parent
            self.data_path = base_path / "monroe county car crach 2003-2015.csv"
        
        self.df = pd.read_csv(self.data_path, encoding='latin-1')
        print(f"Loaded {len(self.df)} records from dataset")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and standardizing categories.
        
        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            self.load_data()
        
        # Create a copy to avoid modifying original
        df = self.df.copy()
        
        # Handle missing values in Primary Factor
        df['Primary Factor'] = df['Primary Factor'].fillna('UNKNOWN')
        df['Primary Factor'] = df['Primary Factor'].replace('', 'UNKNOWN')
        
        # Handle missing Collision Type
        df['Collision Type'] = df['Collision Type'].fillna('Unknown')
        df['Collision Type'] = df['Collision Type'].replace('', 'Unknown')
        
        # Clean Weekend column
        df['Weekend?'] = df['Weekend?'].fillna('Unknown')
        
        # Convert Hour to numeric, handling any errors
        df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce').fillna(0).astype(int)
        
        # Standardize Injury Type categories
        injury_mapping = {
            'No injury/unknown': 'No Injury',
            'Non-incapacitating': 'Minor Injury',
            'Incapacitating': 'Serious Injury',
            'Fatal': 'Fatal'
        }
        df['Injury Type'] = df['Injury Type'].map(injury_mapping).fillna('No Injury')
        
        self.df = df
        return df
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Returns:
            DataFrame with engineered features
        """
        if self.df is None:
            self.clean_data()
        
        df = self.df.copy()
        
        # Time-based features
        df['Hour_Numeric'] = df['Hour'].apply(self._extract_hour)
        df['Time_Period'] = df['Hour_Numeric'].apply(self._categorize_time)
        df['Is_Weekend'] = df['Weekend?'].apply(lambda x: 1 if x == 'Weekend' else 0)
        df['Is_Rush_Hour'] = df['Hour_Numeric'].apply(
            lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
        )
        
        # Categorize primary factors into risk groups
        df['Risk_Factor_Category'] = df['Primary Factor'].apply(self._categorize_risk_factor)
        
        # Collision severity indicator
        collision_severity = {
            '1-Car': 1,
            '2-Car': 2,
            '3+ Cars': 3,
            'Pedestrian': 4,
            'Cyclist': 4,
            'Moped/Motorcycle': 3,
            'Bus': 3,
            'Unknown': 2
        }
        df['Collision_Severity'] = df['Collision Type'].map(collision_severity).fillna(2)
        
        self.df = df
        return df
    
    def _extract_hour(self, hour_val) -> int:
        """Extract hour from various formats."""
        try:
            hour = int(hour_val)
            if hour >= 100:  # Format like 1500 for 3pm
                return hour // 100
            return hour
        except (ValueError, TypeError):
            return 12  # Default to noon
    
    def _categorize_time(self, hour: int) -> str:
        """Categorize hour into time period."""
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    def _categorize_risk_factor(self, factor: str) -> str:
        """Categorize primary factors into risk groups."""
        high_risk = ['UNSAFE SPEED', 'SPEED TOO FAST', 'LEFT OF CENTER', 
                     'DISREGARD SIGNAL', 'RAN OFF ROAD', 'ALCOHOL', 'DRUG']
        medium_risk = ['FAILURE TO YIELD', 'FOLLOWING TOO CLOSELY', 
                       'IMPROPER TURNING', 'IMPROPER LANE']
        
        factor_upper = str(factor).upper()
        
        for risk in high_risk:
            if risk in factor_upper:
                return 'High Risk'
        
        for risk in medium_risk:
            if risk in factor_upper:
                return 'Medium Risk'
        
        return 'Low Risk'
    
    def get_feature_matrix(self) -> tuple:
        """
        Prepare feature matrix and target variable for model training.
        
        Returns:
            Tuple of (X features DataFrame, y target Series)
        """
        if 'Time_Period' not in self.df.columns:
            self.engineer_features()
        
        # Select features for modeling
        self.feature_columns = [
            'Collision Type', 'Weekend?', 'Time_Period', 
            'Risk_Factor_Category', 'Is_Rush_Hour', 
            'Collision_Severity', 'Hour_Numeric'
        ]
        
        X = self.df[self.feature_columns].copy()
        y = self.df['Injury Type'].copy()
        
        return X, y
    
    def get_statistics(self) -> dict:
        """
        Get summary statistics for the dashboard.
        
        Returns:
            Dictionary with various statistics
        """
        if self.df is None:
            self.clean_data()
        
        stats = {
            'total_accidents': len(self.df),
            'injury_distribution': self.df['Injury Type'].value_counts().to_dict(),
            'collision_types': self.df['Collision Type'].value_counts().to_dict(),
            'top_factors': self.df['Primary Factor'].value_counts().head(10).to_dict(),
            'yearly_counts': self.df.groupby('Year').size().to_dict(),
            'weekend_ratio': (self.df['Weekend?'] == 'Weekend').mean() * 100
        }
        
        return stats
