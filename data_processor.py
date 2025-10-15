"""
File 2: data_processor.py
Data processing and categorization for IoT Smart Home data
"""

import pandas as pd
import numpy as np

class SmartHomeDataProcessor:
    def __init__(self):
        """Initialize data categories and normal ranges"""
        
        # Category 1: Device Usage Data (Binary/Status indicators)
        self.device_usage_columns = [
            'Dishwasher', 'Home office', 'Fridge', 'Wine cellar',
            'Garage door', 'Barn', 'Well', 'Microwave', 'Furnace',
            'Kitchen', 'use_HO', 'gen_Sol'
        ]
        
        # Category 2: Energy Consumption Data (Power in kW)
        self.energy_consumption_columns = [
            'Car charger [kW]', 'Water heater [kW]', 'Air conditioning [kW]',
            'Home Theater [kW]', 'Outdoor lights [kW]', 'microwave [kW]',
            'Laundry [kW]', 'Pool Pump [kW]'
        ]
        
        # Category 3: Environmental Data
        self.environmental_columns = [
            'Living room', 'temperature', 'humidity', 'visibility',
            'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover',
            'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability'
        ]
        
        # Normal ranges for different categories
        self.normal_ranges = {
            # Device Usage (0 = OFF, 1 = ON)
            'Dishwasher': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Home office': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Fridge': {'min': 0, 'max': 1, 'description': 'Always ON (should be 1)'},
            'Wine cellar': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Garage door': {'min': 0, 'max': 1, 'description': 'Door Status (0=Closed, 1=Open)'},
            'Barn': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Well': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Microwave': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Furnace': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            'Kitchen': {'min': 0, 'max': 1, 'description': 'Device Status (0=OFF, 1=ON)'},
            
            # Energy Consumption (in kW)
            'Car charger [kW]': {'min': 0, 'max': 7.5, 'description': 'Normal: 0-7.5 kW, Anomaly: >7.5 kW'},
            'Water heater [kW]': {'min': 0, 'max': 4.5, 'description': 'Normal: 0-4.5 kW, Anomaly: >4.5 kW'},
            'Air conditioning [kW]': {'min': 0, 'max': 3.5, 'description': 'Normal: 0-3.5 kW, Anomaly: >3.5 kW'},
            'Home Theater [kW]': {'min': 0, 'max': 0.5, 'description': 'Normal: 0-0.5 kW, Anomaly: >0.5 kW'},
            'Outdoor lights [kW]': {'min': 0, 'max': 0.3, 'description': 'Normal: 0-0.3 kW, Anomaly: >0.3 kW'},
            'microwave [kW]': {'min': 0, 'max': 1.5, 'description': 'Normal: 0-1.5 kW, Anomaly: >1.5 kW'},
            'Laundry [kW]': {'min': 0, 'max': 2.5, 'description': 'Normal: 0-2.5 kW, Anomaly: >2.5 kW'},
            'Pool Pump [kW]': {'min': 0, 'max': 2.0, 'description': 'Normal: 0-2.0 kW, Anomaly: >2.0 kW'},
            
            # Environmental Data
            'Living room': {'min': 0, 'max': 100, 'description': 'Room usage/activity (0-100)'},
            'temperature': {'min': -10, 'max': 40, 'description': 'Normal: 15-30°C, Anomaly: <15 or >30°C'},
            'humidity': {'min': 0, 'max': 100, 'description': 'Normal: 30-70%, Anomaly: <30% or >70%'},
            'visibility': {'min': 0, 'max': 16, 'description': 'Visibility in km (0-16)'},
            'apparentTemperature': {'min': -15, 'max': 45, 'description': 'Feels-like temperature'},
            'pressure': {'min': 950, 'max': 1050, 'description': 'Normal: 980-1020 hPa'},
            'windSpeed': {'min': 0, 'max': 50, 'description': 'Normal: 0-20 km/h, High: >20 km/h'},
            'cloudCover': {'min': 0, 'max': 1, 'description': 'Cloud coverage (0=clear, 1=overcast)'},
            'windBearing': {'min': 0, 'max': 360, 'description': 'Wind direction in degrees'},
            'precipIntensity': {'min': 0, 'max': 10, 'description': 'Precipitation intensity mm/h'},
            'dewPoint': {'min': -20, 'max': 30, 'description': 'Dew point temperature'},
            'precipProbability': {'min': 0, 'max': 1, 'description': 'Rain probability (0-1)'}
        }
    
    def load_dataset(self, file_path):
        """Load and validate dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def categorize_data(self, df):
        """Categorize data into different groups"""
        categories = {}
        
        # Extract Device Usage Data
        device_cols = [col for col in self.device_usage_columns if col in df.columns]
        if device_cols:
            categories['Device Usage'] = df[device_cols].copy()
        
        # Extract Energy Consumption Data
        energy_cols = [col for col in self.energy_consumption_columns if col in df.columns]
        if energy_cols:
            categories['Energy Consumption'] = df[energy_cols].copy()
        
        # Extract Environmental Data
        env_cols = [col for col in self.environmental_columns if col in df.columns]
        if env_cols:
            categories['Environmental'] = df[env_cols].copy()
        
        return categories
    
    def get_statistics(self, df, category_name):
        """Calculate statistics for a category"""
        stats = {
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'median': df.median()
        }
        
        return stats
    
    def check_anomalies_by_range(self, df):
        """Check for anomalies based on predefined ranges"""
        anomalies = {}
        
        for column in df.columns:
            if column in self.normal_ranges:
                range_info = self.normal_ranges[column]
                
                # Find values outside normal range
                out_of_range = df[
                    (df[column] < range_info['min']) | 
                    (df[column] > range_info['max'])
                ]
                
                if len(out_of_range) > 0:
                    anomalies[column] = {
                        'count': len(out_of_range),
                        'percentage': (len(out_of_range) / len(df)) * 100,
                        'indices': out_of_range.index.tolist()
                    }
        
        return anomalies
    
    def get_normal_range_description(self, column):
        """Get normal range description for a column"""
        if column in self.normal_ranges:
            return self.normal_ranges[column]['description']
        return "No range information available"
    
    def preprocess_for_model(self, df):
        """Preprocess data for anomaly detection models"""
        # Select numerical columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove unnecessary columns
        exclude_cols = ['Unnamed: 0', 'year', 'month', 'day', 'weekday', 'weekofyear', 'hour', 'minute']
        feature_columns = [col for col in numeric_columns if col not in exclude_cols]
        
        # Extract features
        df_features = df[feature_columns].copy()
        
        # Handle missing values
        df_features = df_features.fillna(df_features.mean())
        
        return df_features, feature_columns
    
    def generate_report(self, df):
        """Generate comprehensive data report"""
        report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'categories': {}
        }
        
        # Categorize and get stats for each category
        categories = self.categorize_data(df)
        
        for cat_name, cat_df in categories.items():
            report['categories'][cat_name] = {
                'columns': cat_df.columns.tolist(),
                'statistics': self.get_statistics(cat_df, cat_name).to_dict()
            }
        
        return report