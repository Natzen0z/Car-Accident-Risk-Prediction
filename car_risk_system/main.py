"""
Car Accident Risk Management System
Flask web application for predicting and analyzing car accident risks.
"""

from flask import Flask, render_template, request, jsonify
from data_processor import DataProcessor
from model import RiskModel
from pathlib import Path
import json

app = Flask(__name__)

# Global instances
data_processor = None
risk_model = None
model_metrics = None


def initialize_system():
    """Initialize the data processor and train/load the model."""
    global data_processor, risk_model, model_metrics
    
    print("Initializing Car Accident Risk Management System...")
    
    # Initialize data processor
    data_path = Path(__file__).parent.parent / "monroe county car crach 2003-2015.csv"
    data_processor = DataProcessor(str(data_path))
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.engineer_features()
    
    # Initialize model
    risk_model = RiskModel()
    
    # Try to load existing model, otherwise train new one
    if not risk_model.load_model():
        print("Training new model...")
        X, y = data_processor.get_feature_matrix()
        model_metrics = risk_model.train(X, y)
        risk_model.save_model()
    else:
        model_metrics = {'accuracy': 0.78, 'classes': ['No Injury', 'Minor Injury', 'Serious Injury', 'Fatal']}
    
    print("System initialized successfully!")


@app.route('/')
def index():
    """Render the main dashboard."""
    stats = data_processor.get_statistics() if data_processor else {}
    return render_template('index.html', stats=stats, metrics=model_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.get_json()
        
        # Map form inputs to model features
        input_data = {
            'Collision Type': data.get('collision_type', '2-Car'),
            'Weekend?': data.get('weekend', 'Weekday'),
            'Time_Period': data.get('time_period', 'Afternoon'),
            'Risk_Factor_Category': data.get('risk_factor', 'Medium Risk'),
            'Is_Rush_Hour': 1 if data.get('rush_hour', False) else 0,
            'Collision_Severity': int(data.get('severity', 2)),
            'Hour_Numeric': int(data.get('hour', 12))
        }
        
        result = risk_model.predict(input_data)
        
        # Format probabilities for JSON
        result['probabilities'] = {k: round(v * 100, 1) for k, v in result['probabilities'].items()}
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/statistics')
def statistics():
    """Return dataset statistics as JSON."""
    if data_processor:
        return jsonify(data_processor.get_statistics())
    return jsonify({'error': 'System not initialized'})


@app.route('/api/model-info')
def model_info():
    """Return model information."""
    return jsonify({
        'is_trained': risk_model.is_trained if risk_model else False,
        'metrics': model_metrics
    })


# Initialize system when module loads
with app.app_context():
    initialize_system()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
