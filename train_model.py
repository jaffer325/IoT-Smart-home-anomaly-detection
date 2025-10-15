"""
File 1: train_model.py
Training script for IoT Smart Home Anomaly Detection
This file trains Autoencoder and LSTM models on normal smart home data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

warnings.filterwarnings('ignore')

class SmartHomeTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.lstm_model = None
        self.feature_columns = None
        self.sequence_length = 10
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the training data"""
        print("Loading training data...")
        df = pd.read_csv(file_path)
        
        # Select numerical features for training
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove unnecessary columns
        exclude_cols = ['Unnamed: 0', 'year', 'month', 'day', 'weekday', 'weekofyear', 'hour', 'minute']
        self.feature_columns = [col for col in numeric_columns if col not in exclude_cols]
        
        print(f"Selected {len(self.feature_columns)} features for training")
        
        # Handle missing values
        df_clean = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        
        # Normalize the data
        data_normalized = self.scaler.fit_transform(df_clean)
        
        return data_normalized
    
    def build_autoencoder(self, input_dim):
        """Build Autoencoder model for anomaly detection"""
        print("Building Autoencoder model...")
        
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation='relu')(input_layer)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time-series anomaly detection"""
        print("Building LSTM model...")
        
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            RepeatVector(input_shape[0]),
            LSTM(32, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(input_shape[1]))
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def train_models(self, file_path, epochs=50, batch_size=32):
        """Train both Autoencoder and LSTM models"""
        # Load and preprocess data
        data_normalized = self.load_and_preprocess_data(file_path)
        
        # Train Autoencoder
        print("\n=== Training Autoencoder ===")
        self.autoencoder = self.build_autoencoder(data_normalized.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history_ae = self.autoencoder.fit(
            data_normalized, data_normalized,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Calculate reconstruction error threshold (95th percentile)
        reconstructions = self.autoencoder.predict(data_normalized)
        mse = np.mean(np.power(data_normalized - reconstructions, 2), axis=1)
        self.ae_threshold = np.percentile(mse, 95)
        print(f"Autoencoder threshold (95th percentile): {self.ae_threshold:.6f}")
        
        # Train LSTM
        print("\n=== Training LSTM ===")
        sequences = self.create_sequences(data_normalized)
        
        self.lstm_model = self.build_lstm_model((self.sequence_length, data_normalized.shape[1]))
        
        history_lstm = self.lstm_model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Calculate LSTM reconstruction error threshold
        lstm_reconstructions = self.lstm_model.predict(sequences)
        lstm_mse = np.mean(np.power(sequences - lstm_reconstructions, 2), axis=(1, 2))
        self.lstm_threshold = np.percentile(lstm_mse, 95)
        print(f"LSTM threshold (95th percentile): {self.lstm_threshold:.6f}")
        
        return history_ae, history_lstm
    
    def save_models(self, model_dir='models'):
        """Save trained models and preprocessing objects"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        print("\nSaving models...")
        
        # Save Autoencoder
        self.autoencoder.save(f'{model_dir}/autoencoder_model.h5')
        
        # Save LSTM
        self.lstm_model.save(f'{model_dir}/lstm_model.h5')
        
        # Save scaler and other parameters
        model_params = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'ae_threshold': self.ae_threshold,
            'lstm_threshold': self.lstm_threshold,
            'sequence_length': self.sequence_length
        }
        
        with open(f'{model_dir}/model_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)
        
        print(f"Models saved to '{model_dir}/' directory")
        print(f"- autoencoder_model.h5")
        print(f"- lstm_model.h5")
        print(f"- model_params.pkl")

if __name__ == "__main__":
    # Initialize trainer
    trainer = SmartHomeTrainer()
    
    # Train models on normal data
    print("=" * 60)
    print("IoT Smart Home Anomaly Detection - Model Training")
    print("=" * 60)
    
    training_file = "oneday_home.csv"  # Your training dataset
    
    try:
        history_ae, history_lstm = trainer.train_models(
            training_file,
            epochs=50,
            batch_size=32
        )
        
        # Save models
        trainer.save_models()
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please ensure 'oneday_home.csv' is in the same directory")