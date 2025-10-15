"""
File: evaluate_model_unsupervised.py
Unsupervised evaluation of Autoencoder and LSTM models for Smart Home Anomaly Detection
Calculates reconstruction error statistics and anomaly ratios.
"""

import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class SmartHomeUnsupervisedEvaluator:
    def __init__(self, model_dir='models'):
        # Load model parameters
        with open(f'{model_dir}/model_params.pkl', 'rb') as f:
            params = pickle.load(f)
        
        self.scaler = params['scaler']
        self.feature_columns = params['feature_columns']
        self.ae_threshold = params['ae_threshold']
        self.lstm_threshold = params['lstm_threshold']
        self.sequence_length = params['sequence_length']

        # Load trained models
        self.autoencoder = load_model(f'{model_dir}/autoencoder_model.h5')
        self.lstm_model = load_model(f'{model_dir}/lstm_model.h5')

    def preprocess_data(self, file_path):
        df = pd.read_csv(file_path)
        df = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        data_scaled = self.scaler.transform(df)
        return data_scaled

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)

    def evaluate_autoencoder(self, X_test):
        reconstructions = self.autoencoder.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

        # Identify anomalies
        y_pred = (mse > self.ae_threshold).astype(int)

        print("\n=== Autoencoder Evaluation ===")
        print(f"Reconstruction Error - Mean: {np.mean(mse):.6f}, Std: {np.std(mse):.6f}")
        print(f"95th Percentile (Threshold): {self.ae_threshold:.6f}")
        print(f"Detected Anomalies: {np.sum(y_pred)} / {len(y_pred)} ({np.mean(y_pred)*100:.2f}%)")

        # Plot reconstruction error
        plt.figure(figsize=(8, 4))
        plt.hist(mse, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(self.ae_threshold, color='red', linestyle='--', label='Threshold')
        plt.title("Autoencoder Reconstruction Error Distribution")
        plt.xlabel("Reconstruction Error (MSE)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def evaluate_lstm(self, X_test):
        sequences = self.create_sequences(X_test)
        reconstructions = self.lstm_model.predict(sequences)
        mse = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))

        y_pred = (mse > self.lstm_threshold).astype(int)

        print("\n=== LSTM Evaluation ===")
        print(f"Reconstruction Error - Mean: {np.mean(mse):.6f}, Std: {np.std(mse):.6f}")
        print(f"95th Percentile (Threshold): {self.lstm_threshold:.6f}")
        print(f"Detected Anomalies: {np.sum(y_pred)} / {len(y_pred)} ({np.mean(y_pred)*100:.2f}%)")

        # Plot reconstruction error
        plt.figure(figsize=(8, 4))
        plt.hist(mse, bins=50, color='lightgreen', edgecolor='black')
        plt.axvline(self.lstm_threshold, color='red', linestyle='--', label='Threshold')
        plt.title("LSTM Reconstruction Error Distribution")
        plt.xlabel("Reconstruction Error (MSE)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    test_file = "oneday_home.csv"  # Unlabeled test data

    evaluator = SmartHomeUnsupervisedEvaluator()
    data_scaled = evaluator.preprocess_data(test_file)

    evaluator.evaluate_autoencoder(data_scaled)
    evaluator.evaluate_lstm(data_scaled)
