"""
File 3: visualization.py
Visualization and anomaly detection using Autoencoder and LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle

class SmartHomeVisualizer:
    def __init__(self):
        self.models_loaded = False
        self.autoencoder = None
        self.lstm_model = None
        self.scaler = None
        self.feature_columns = None
        self.ae_threshold = None
        self.lstm_threshold = None
        self.sequence_length = 10
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def load_trained_models(self, model_dir='models'):
        """Load trained models and parameters"""
        try:
            print("Loading trained models...")
            
            # Load models
            self.autoencoder = load_model(f'{model_dir}/autoencoder_model.h5', compile=False)
            self.lstm_model = load_model(f'{model_dir}/lstm_model.h5', compile=False)
            
            # Load parameters
            with open(f'{model_dir}/model_params.pkl', 'rb') as f:
                params = pickle.load(f)
            
            self.scaler = params['scaler']
            self.feature_columns = params['feature_columns']
            self.ae_threshold = params['ae_threshold']
            self.lstm_threshold = params['lstm_threshold']
            self.sequence_length = params['sequence_length']
            
            self.models_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Please run train_model.py first to train the models.")
            return False
    
    def detect_anomalies_autoencoder(self, df):
        """Detect anomalies using Autoencoder"""
        if not self.models_loaded:
            return None, None, None
        
        # Prepare data
        df_features = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        data_normalized = self.scaler.transform(df_features)
        
        # Get reconstructions
        reconstructions = self.autoencoder.predict(data_normalized, verbose=0)
        
        # Calculate reconstruction errors
        mse = np.mean(np.power(data_normalized - reconstructions, 2), axis=1)
        
        # Identify anomalies
        anomalies = mse > self.ae_threshold
        
        return mse, anomalies, self.ae_threshold
    
    def detect_anomalies_lstm(self, df):
        """Detect anomalies using LSTM"""
        if not self.models_loaded:
            return None, None, None
        
        # Prepare data
        df_features = df[self.feature_columns].fillna(df[self.feature_columns].mean())
        data_normalized = self.scaler.transform(df_features)
        
        # Create sequences
        sequences = []
        for i in range(len(data_normalized) - self.sequence_length):
            sequences.append(data_normalized[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        if len(sequences) == 0:
            return None, None, None
        
        # Get reconstructions
        lstm_reconstructions = self.lstm_model.predict(sequences, verbose=0)
        
        # Calculate reconstruction errors
        lstm_mse = np.mean(np.power(sequences - lstm_reconstructions, 2), axis=(1, 2))
        
        # Identify anomalies
        anomalies = np.zeros(len(data_normalized), dtype=bool)
        anomalies[self.sequence_length:] = lstm_mse > self.lstm_threshold
        
        # Extend error array to match original length
        full_mse = np.zeros(len(data_normalized))
        full_mse[self.sequence_length:] = lstm_mse
        
        return full_mse, anomalies, self.lstm_threshold
    
    def plot_category_overview(self, df, category_data, category_name, canvas_widget):
        """Plot overview of a data category"""
        fig = Figure(figsize=(12, 8), dpi=100)
        
        if category_data.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'No data available for {category_name}', 
                   ha='center', va='center', fontsize=14)
            self._embed_figure(fig, canvas_widget)
            return
        
        n_cols = min(3, len(category_data.columns))
        n_rows = (len(category_data.columns) + n_cols - 1) // n_cols
        
        for idx, column in enumerate(category_data.columns, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx)
            
            data = category_data[column].dropna()
            
            if len(data) > 0:
                ax.plot(data.index, data.values, linewidth=1, alpha=0.7, color='steelblue')
                ax.fill_between(data.index, data.values, alpha=0.3, color='steelblue')
                ax.set_title(column, fontsize=10, fontweight='bold')
                ax.set_xlabel('Time Index', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
        
        fig.suptitle(f'{category_name} Overview', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        self._embed_figure(fig, canvas_widget)
    
    def plot_anomaly_detection_results(self, df, canvas_widget):
        """Plot comprehensive anomaly detection results"""
        if not self.models_loaded:
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Models not loaded. Please train models first.', 
                   ha='center', va='center', fontsize=14, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        # Detect anomalies with both methods
        ae_mse, ae_anomalies, ae_thresh = self.detect_anomalies_autoencoder(df)
        lstm_mse, lstm_anomalies, lstm_thresh = self.detect_anomalies_lstm(df)
        
        if ae_mse is None or lstm_mse is None:
            fig = Figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Error in anomaly detection', 
                   ha='center', va='center', fontsize=14, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        # Create comprehensive visualization
        fig = Figure(figsize=(14, 10), dpi=100)
        
        # 1. Autoencoder Reconstruction Error
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(ae_mse, label='Reconstruction Error', linewidth=1, color='blue', alpha=0.7)
        ax1.axhline(y=ae_thresh, color='r', linestyle='--', label=f'Threshold: {ae_thresh:.4f}')
        ax1.fill_between(range(len(ae_mse)), ae_mse, ae_thresh, 
                         where=(ae_mse > ae_thresh), color='red', alpha=0.3, label='Anomalies')
        ax1.set_title('Autoencoder: Reconstruction Error', fontweight='bold')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('MSE')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. LSTM Reconstruction Error
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(lstm_mse, label='Reconstruction Error', linewidth=1, color='green', alpha=0.7)
        ax2.axhline(y=lstm_thresh, color='r', linestyle='--', label=f'Threshold: {lstm_thresh:.4f}')
        ax2.fill_between(range(len(lstm_mse)), lstm_mse, lstm_thresh,
                         where=(lstm_mse > lstm_thresh), color='red', alpha=0.3, label='Anomalies')
        ax2.set_title('LSTM: Reconstruction Error', fontweight='bold')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('MSE')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Anomaly Distribution - Autoencoder
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.hist(ae_mse, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.axvline(x=ae_thresh, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax3.set_title('Autoencoder: Error Distribution', fontweight='bold')
        ax3.set_xlabel('Reconstruction Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Anomaly Distribution - LSTM
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.hist(lstm_mse[lstm_mse > 0], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(x=lstm_thresh, color='r', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_title('LSTM: Error Distribution', fontweight='bold')
        ax4.set_xlabel('Reconstruction Error')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Combined Anomaly Timeline
        ax5 = fig.add_subplot(3, 1, 3)
        
        # Create timeline
        timeline = np.zeros(len(ae_anomalies))
        timeline[ae_anomalies] = 1
        timeline[lstm_anomalies] = 2
        timeline[(ae_anomalies) & (lstm_anomalies)] = 3
        
        colors = ['green', 'orange', 'yellow', 'red']
        labels = ['Normal', 'AE Anomaly', 'LSTM Anomaly', 'Both Detected']
        
        for i in range(len(timeline)):
            ax5.axvspan(i, i+1, alpha=0.5, color=colors[int(timeline[i])])
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.5, label=labels[i]) for i in range(4)]
        ax5.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        ax5.set_title('Combined Anomaly Detection Timeline', fontweight='bold')
        ax5.set_xlabel('Time Index')
        ax5.set_ylabel('Detection Status')
        ax5.set_yticks([])
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Add statistics
        ae_count = np.sum(ae_anomalies)
        lstm_count = np.sum(lstm_anomalies)
        both_count = np.sum((ae_anomalies) & (lstm_anomalies))
        
        stats_text = f'Autoencoder: {ae_count} anomalies ({ae_count/len(ae_anomalies)*100:.1f}%)\n'
        stats_text += f'LSTM: {lstm_count} anomalies ({lstm_count/len(lstm_anomalies)*100:.1f}%)\n'
        stats_text += f'Both Methods: {both_count} anomalies ({both_count/len(ae_anomalies)*100:.1f}%)'
        
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle('IoT Smart Home - Anomaly Detection Results', 
                    fontsize=16, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        self._embed_figure(fig, canvas_widget)
        
        # Return anomaly information
        return {
            'ae_anomalies': ae_anomalies,
            'lstm_anomalies': lstm_anomalies,
            'ae_count': ae_count,
            'lstm_count': lstm_count,
            'both_count': both_count,
            'total_records': len(ae_anomalies)
        }
    
    def plot_energy_consumption_analysis(self, df, energy_data, canvas_widget):
        """Plot detailed energy consumption analysis"""
        fig = Figure(figsize=(14, 10), dpi=100)
        
        if energy_data.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No energy consumption data available', 
                   ha='center', va='center', fontsize=14)
            self._embed_figure(fig, canvas_widget)
            return
        
        # 1. Total energy consumption over time
        ax1 = fig.add_subplot(3, 2, 1)
        total_energy = energy_data.sum(axis=1)
        ax1.plot(total_energy.index, total_energy.values, linewidth=1.5, color='darkblue')
        ax1.fill_between(total_energy.index, total_energy.values, alpha=0.3, color='lightblue')
        ax1.set_title('Total Energy Consumption Over Time', fontweight='bold')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Total Power (kW)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Average consumption per device
        ax2 = fig.add_subplot(3, 2, 2)
        avg_consumption = energy_data.mean().sort_values(ascending=False)
        colors_bar = plt.cm.viridis(np.linspace(0, 1, len(avg_consumption)))
        ax2.barh(range(len(avg_consumption)), avg_consumption.values, color=colors_bar)
        ax2.set_yticks(range(len(avg_consumption)))
        ax2.set_yticklabels([col.replace(' [kW]', '') for col in avg_consumption.index], fontsize=8)
        ax2.set_xlabel('Average Power (kW)')
        ax2.set_title('Average Energy Consumption by Device', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Energy consumption heatmap
        ax3 = fig.add_subplot(3, 2, 3)
        # Sample data if too large
        sample_data = energy_data.iloc[::max(1, len(energy_data)//100)]
        sns.heatmap(sample_data.T, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Power (kW)'})
        ax3.set_title('Energy Consumption Heatmap', fontweight='bold')
        ax3.set_xlabel('Time Index (sampled)')
        ax3.set_ylabel('Devices')
        ax3.set_yticklabels([col.replace(' [kW]', '') for col in sample_data.columns], 
                           rotation=0, fontsize=8)
        
        # 4. Peak consumption analysis
        ax4 = fig.add_subplot(3, 2, 4)
        max_consumption = energy_data.max().sort_values(ascending=False)
        colors_bar2 = plt.cm.plasma(np.linspace(0, 1, len(max_consumption)))
        ax4.barh(range(len(max_consumption)), max_consumption.values, color=colors_bar2)
        ax4.set_yticks(range(len(max_consumption)))
        ax4.set_yticklabels([col.replace(' [kW]', '') for col in max_consumption.index], fontsize=8)
        ax4.set_xlabel('Peak Power (kW)')
        ax4.set_title('Peak Energy Consumption by Device', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Consumption distribution
        ax5 = fig.add_subplot(3, 2, 5)
        for column in energy_data.columns[:4]:  # Top 4 devices
            data = energy_data[column][energy_data[column] > 0]
            if len(data) > 0:
                ax5.hist(data, bins=30, alpha=0.5, label=column.replace(' [kW]', ''), edgecolor='black')
        ax5.set_xlabel('Power (kW)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Energy Consumption Distribution (Top 4 Devices)', fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.axis('off')
        
        total_avg = total_energy.mean()
        total_max = total_energy.max()
        total_std = total_energy.std()
        
        stats_text = "Energy Consumption Summary\n" + "="*40 + "\n\n"
        stats_text += f"Average Total Power: {total_avg:.2f} kW\n"
        stats_text += f"Peak Total Power: {total_max:.2f} kW\n"
        stats_text += f"Std Deviation: {total_std:.2f} kW\n\n"
        stats_text += "Top Energy Consumers:\n"
        for device, value in avg_consumption.head(5).items():
            stats_text += f"  • {device.replace(' [kW]', '')}: {value:.2f} kW\n"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle('Energy Consumption Analysis', fontsize=16, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        self._embed_figure(fig, canvas_widget)
    
    def plot_device_usage_analysis(self, device_data, canvas_widget):
        """Plot device usage patterns"""
        fig = Figure(figsize=(12, 8), dpi=100)
        
        if device_data.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No device usage data available', 
                   ha='center', va='center', fontsize=14)
            self._embed_figure(fig, canvas_widget)
            return
        
        # 1. Device usage over time
        ax1 = fig.add_subplot(2, 2, 1)
        for column in device_data.columns[:5]:  # Show first 5 devices
            ax1.plot(device_data.index, device_data[column], 
                    label=column, linewidth=1, alpha=0.7)
        ax1.set_title('Device Usage Timeline (First 5 Devices)', fontweight='bold')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Status (0=OFF, 1=ON)')
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # 2. Usage frequency
        ax2 = fig.add_subplot(2, 2, 2)
        usage_freq = (device_data.sum() / len(device_data) * 100).sort_values(ascending=False)
        colors_bar = plt.cm.coolwarm(np.linspace(0, 1, len(usage_freq)))
        ax2.barh(range(len(usage_freq)), usage_freq.values, color=colors_bar)
        ax2.set_yticks(range(len(usage_freq)))
        ax2.set_yticklabels(usage_freq.index, fontsize=8)
        ax2.set_xlabel('Usage Frequency (%)')
        ax2.set_title('Device Usage Frequency', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Correlation heatmap
        ax3 = fig.add_subplot(2, 2, 3)
        corr_matrix = device_data.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   ax=ax3, cbar_kws={'label': 'Correlation'})
        ax3.set_title('Device Usage Correlation', fontweight='bold')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0, fontsize=7)
        
        # 4. Summary
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        summary_text = "Device Usage Summary\n" + "="*35 + "\n\n"
        summary_text += f"Total Devices Monitored: {len(device_data.columns)}\n"
        summary_text += f"Total Time Points: {len(device_data)}\n\n"
        summary_text += "Most Active Devices:\n"
        for device, freq in usage_freq.head(5).items():
            summary_text += f"  • {device}: {freq:.1f}% active\n"
        summary_text += "\nLeast Active Devices:\n"
        for device, freq in usage_freq.tail(3).items():
            summary_text += f"  • {device}: {freq:.1f}% active\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        fig.suptitle('Device Usage Analysis', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        
        self._embed_figure(fig, canvas_widget)
    
    def _embed_figure(self, fig, canvas_widget):
        """Embed matplotlib figure in tkinter canvas"""
        for widget in canvas_widget.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)