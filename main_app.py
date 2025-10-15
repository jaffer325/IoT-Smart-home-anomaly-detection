"""
File: main_app.py
Enhanced Anomaly Detection with Twilio SMS Alerts
Simplified UI - credentials loaded from twilio_alert.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from tensorflow.keras.models import load_model
import pickle
from datetime import datetime

# Import custom modules
from data_processor import SmartHomeDataProcessor
from twilio_alert import TwilioAnomalyAlert, get_twilio_config


class AnomalyFocusedVisualizer:
    def __init__(self):
        self.models_loaded = False
        self.autoencoder = None
        self.lstm_model = None
        self.scaler = None
        self.feature_columns = None
        self.ae_threshold = None
        self.lstm_threshold = None
        self.sequence_length = 10
        
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def load_trained_models(self, model_dir='models'):
        """Load trained models and parameters"""
        try:
            print("Loading trained models...")
            
            self.autoencoder = load_model(f'{model_dir}/autoencoder_model.h5', compile=False)
            self.lstm_model = load_model(f'{model_dir}/lstm_model.h5', compile=False)
            
            with open(f'{model_dir}/model_params.pkl', 'rb') as f:
                params = pickle.load(f)
            
            self.scaler = params['scaler']
            self.feature_columns = params['feature_columns']
            self.ae_threshold = params['ae_threshold']
            self.lstm_threshold = params['lstm_threshold']
            self.sequence_length = params['sequence_length']
            
            self.models_loaded = True
            print("✓ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def detect_anomalies(self, df):
        """Detect anomalies using both models"""
        if not self.models_loaded:
            return None
        
        available_features = [col for col in self.feature_columns if col in df.columns]
        
        if len(available_features) == 0:
            print("Error: No matching features found in test data!")
            return None
        
        df_features = df[available_features].fillna(df[available_features].mean())
        
        if len(available_features) < len(self.feature_columns):
            full_features = np.zeros((len(df_features), len(self.feature_columns)))
            for i, feat in enumerate(self.feature_columns):
                if feat in available_features:
                    idx = available_features.index(feat)
                    full_features[:, i] = df_features.iloc[:, idx].values
            data_normalized = self.scaler.transform(full_features)
        else:
            data_normalized = self.scaler.transform(df_features)
        
        # Autoencoder detection
        ae_reconstructions = self.autoencoder.predict(data_normalized, verbose=0)
        ae_mse = np.mean(np.power(data_normalized - ae_reconstructions, 2), axis=1)
        ae_anomalies = ae_mse > self.ae_threshold
        
        # LSTM detection
        sequences = []
        for i in range(len(data_normalized) - self.sequence_length):
            sequences.append(data_normalized[i:i + self.sequence_length])
        sequences = np.array(sequences)
        
        lstm_anomalies = np.zeros(len(data_normalized), dtype=bool)
        if len(sequences) > 0:
            lstm_reconstructions = self.lstm_model.predict(sequences, verbose=0)
            lstm_mse = np.mean(np.power(sequences - lstm_reconstructions, 2), axis=(1, 2))
            lstm_anomalies[self.sequence_length:] = lstm_mse > self.lstm_threshold
        
        combined_anomalies = ae_anomalies | lstm_anomalies
        
        return {
            'ae_anomalies': ae_anomalies,
            'lstm_anomalies': lstm_anomalies,
            'combined_anomalies': combined_anomalies,
            'ae_mse': ae_mse,
            'anomaly_indices': np.where(combined_anomalies)[0]
        }
    
    def plot_multi_feature_timeline(self, df, selected_features, canvas_widget, data_processor):
        """Plot multi-feature timeline with anomaly detection"""
        if not self.models_loaded:
            fig = Figure(figsize=(14, 10))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Models not loaded. Please train models first.', 
                   ha='center', va='center', fontsize=14, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        try:
            anomaly_results = self.detect_anomalies(df)
            if anomaly_results is None:
                fig = Figure(figsize=(14, 10))
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Error: No matching features found in test data!', 
                       ha='center', va='center', fontsize=12, color='red')
                self._embed_figure(fig, canvas_widget)
                return None
        except Exception as e:
            fig = Figure(figsize=(14, 10))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Error during anomaly detection:\n{str(e)}', 
                   ha='center', va='center', fontsize=12, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        anomaly_indices = anomaly_results['anomaly_indices']
        
        available_features = [f for f in selected_features if f in df.columns]
        if not available_features:
            available_features = [f for f in df.columns if f in self.feature_columns][:5]
        
        plot_features = available_features[:6]
        
        fig = Figure(figsize=(14, 10), dpi=100)
        n_features = len(plot_features)
        
        for idx, feature in enumerate(plot_features, 1):
            ax = fig.add_subplot(n_features, 1, idx)
            
            data = df[feature].values
            x_indices = np.arange(len(data))
            
            ax.plot(x_indices, data, linewidth=1, color='steelblue', alpha=0.7, label=feature)
            
            if len(anomaly_indices) > 0:
                anomaly_mask = np.zeros(len(data), dtype=bool)
                anomaly_mask[anomaly_indices] = True
                ax.scatter(x_indices[anomaly_mask], data[anomaly_mask], 
                          color='red', s=30, marker='o', label='Anomaly', 
                          zorder=5, alpha=0.8, edgecolors='darkred', linewidths=1)
            
            if feature in data_processor.normal_ranges:
                threshold = data_processor.normal_ranges[feature]['max']
                ax.axhline(y=threshold, color='orange', linestyle='--', 
                          linewidth=1.5, alpha=0.6, label=f'Threshold: {threshold}')
            
            ax.set_ylabel(feature.replace(' [kW]', ''), fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
            ax.tick_params(labelsize=8)
            
            if idx == n_features:
                ax.set_xlabel('Row Index', fontsize=10, fontweight='bold')
            else:
                ax.set_xticklabels([])
        
        anomaly_count = len(anomaly_indices)
        anomaly_pct = (anomaly_count / len(df)) * 100
        fig.suptitle(f'Smart Home Multi-Feature Anomaly Detection\n'
                    f'Total Anomalies: {anomaly_count} ({anomaly_pct:.2f}%)',
                    fontsize=14, fontweight='bold', y=0.995)
        
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._embed_figure(fig, canvas_widget)
        
        return anomaly_results
    
    def plot_single_feature_detailed(self, df, feature, canvas_widget, data_processor):
        """Plot single feature with detailed analysis"""
        if not self.models_loaded:
            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Models not loaded. Please train models first.', 
                   ha='center', va='center', fontsize=14, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        if feature not in df.columns:
            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f'Feature "{feature}" not found in dataset!', 
                   ha='center', va='center', fontsize=14, color='red')
            self._embed_figure(fig, canvas_widget)
            return None
        
        try:
            anomaly_results = self.detect_anomalies(df)
            if anomaly_results is None:
                return None
        except Exception as e:
            return None
        
        fig = Figure(figsize=(12, 9), dpi=100)
        
        data = df[feature].values
        x_indices = np.arange(len(data))
        anomaly_indices = anomaly_results['anomaly_indices']
        anomaly_mask = np.zeros(len(data), dtype=bool)
        if len(anomaly_indices) > 0:
            anomaly_mask[anomaly_indices] = True
        
        # Main timeline
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(x_indices, data, linewidth=1.5, color='steelblue', alpha=0.8)
        
        if len(anomaly_indices) > 0:
            ax1.scatter(x_indices[anomaly_mask], data[anomaly_mask], 
                       color='red', s=50, marker='o', label='Anomaly', 
                       zorder=5, alpha=0.9, edgecolors='darkred', linewidths=1.5)
        
        if feature in data_processor.normal_ranges:
            threshold = data_processor.normal_ranges[feature]['max']
            ax1.axhline(y=threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Threshold: {threshold}')
        
        ax1.set_title(f'{feature} - Anomaly Detection', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Time Index', fontweight='bold')
        ax1.set_ylabel('Value', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Statistics
        ax2 = fig.add_subplot(3, 2, 2)
        normal_data = data[~anomaly_mask]
        anomaly_data = data[anomaly_mask] if len(anomaly_indices) > 0 else np.array([])
        
        stats_labels = ['Mean', 'Median', 'Max']
        normal_stats = [normal_data.mean(), np.median(normal_data), normal_data.max()] if len(normal_data) > 0 else [0, 0, 0]
        anomaly_stats = [anomaly_data.mean(), np.median(anomaly_data), anomaly_data.max()] if len(anomaly_data) > 0 else [0, 0, 0]
        
        x_pos = np.arange(len(stats_labels))
        width = 0.35
        
        ax2.bar(x_pos - width/2, normal_stats, width, label='Normal', color='steelblue', alpha=0.8)
        ax2.bar(x_pos + width/2, anomaly_stats, width, label='Anomaly', color='red', alpha=0.8)
        
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Normal vs Anomaly Statistics', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stats_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Distribution
        ax3 = fig.add_subplot(3, 2, 3)
        if len(normal_data) > 0:
            ax3.hist(normal_data, bins=40, alpha=0.7, color='steelblue', 
                    label='Normal', edgecolor='black', linewidth=0.5)
        if len(anomaly_data) > 0:
            ax3.hist(anomaly_data, bins=20, alpha=0.7, color='red', 
                    label='Anomaly', edgecolor='darkred', linewidth=0.5)
        
        ax3.set_xlabel('Value', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Value Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Anomaly score
        ax4 = fig.add_subplot(3, 2, 4)
        ae_mse = anomaly_results['ae_mse']
        ax4.plot(x_indices, ae_mse, linewidth=1, color='purple', alpha=0.7)
        ax4.axhline(y=self.ae_threshold, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold: {self.ae_threshold:.4f}')
        ax4.fill_between(x_indices, ae_mse, self.ae_threshold,
                        where=(ae_mse > self.ae_threshold), 
                        color='red', alpha=0.3)
        ax4.set_title('Anomaly Score', fontweight='bold')
        ax4.set_xlabel('Time Index', fontweight='bold')
        ax4.set_ylabel('MSE', fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Timeline
        ax5 = fig.add_subplot(3, 1, 3)
        colors = ['green' if not a else 'red' for a in anomaly_mask]
        for i in range(len(data)):
            ax5.axvspan(i, i+1, alpha=0.6, color=colors[i])
        
        ax5.set_title('Anomaly Detection Timeline', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Time Index', fontweight='bold')
        ax5.set_ylabel('Status', fontweight='bold')
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Normal', 'Anomaly'])
        ax5.grid(True, alpha=0.3, axis='x')
        
        anomaly_count = len(anomaly_indices)
        anomaly_pct = (anomaly_count / len(data)) * 100 if len(data) > 0 else 0
        stats_text = f'Anomalies: {anomaly_count}/{len(data)} ({anomaly_pct:.2f}%)\n'
        
        if len(normal_data) > 0:
            stats_text += f'Normal Mean: {normal_data.mean():.3f}\n'
        if len(anomaly_data) > 0:
            stats_text += f'Anomaly Mean: {anomaly_data.mean():.3f}'
        
        ax5.text(0.02, 0.98, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        fig.tight_layout()
        self._embed_figure(fig, canvas_widget)
        
        return anomaly_results
    
    def _embed_figure(self, fig, canvas_widget):
        """Embed figure in canvas"""
        for widget in canvas_widget.winfo_children():
            widget.destroy()
        
        canvas = FigureCanvasTkAgg(fig, master=canvas_widget)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


class AnomalyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IoT Smart Home Anomaly Detection with Twilio Alerts")
        self.root.geometry("1400x900")
        self.root.configure(bg='#ecf0f1')
        
        self.data_processor = SmartHomeDataProcessor()
        self.visualizer = AnomalyFocusedVisualizer()
        
        # Initialize Twilio from config
        try:
            config = get_twilio_config()
            self.alert_system = TwilioAnomalyAlert(
                account_sid=config['account_sid'],
                auth_token=config['auth_token'],
                from_phone=config['from_phone'],
                to_phone=config['to_phone']
            )
            self.twilio_configured = True
        except Exception as e:
            print(f"⚠️ Twilio initialization failed: {str(e)}")
            self.alert_system = None
            self.twilio_configured = False
        
        self.training_df = None
        self.test_df = None
        
        self.feature_groups = {
            'Energy Consumption': [
                'Car charger [kW]', 'Water heater [kW]', 'Air conditioning [kW]',
                'Home Theater [kW]', 'microwave [kW]', 'Laundry [kW]', 'Pool Pump [kW]'
            ],
            'Device Usage': [
                'Dishwasher', 'Fridge', 'Furnace', 'Kitchen', 'Microwave'
            ],
            'Environmental': [
                'temperature', 'humidity', 'pressure', 'windSpeed', 'Living room'
            ]
        }
        
        self.models_loaded = self.visualizer.load_trained_models()
        self.auto_load_training_data()
        
        self.create_gui()
    
    def auto_load_training_data(self):
        """Auto-load training data"""
        training_file = "oneday_home.csv"
        if os.path.exists(training_file):
            try:
                self.training_df = self.data_processor.load_dataset(training_file)
                print("✓ Training data loaded")
            except Exception as e:
                print(f"Could not auto-load training data: {str(e)}")
    
    def create_gui(self):
        """Create GUI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', side='top')
        title_frame.pack_propagate(False)
        
        tk.Label(
            title_frame,
            text="🏠 IoT Smart Home - Anomaly Detection with SMS Alerts",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left Panel
        left_panel_container = tk.Frame(main_container, bg='white', width=320, relief='raised', borderwidth=2)
        left_panel_container.pack(side='left', fill='y', padx=(0, 5))
        left_panel_container.pack_propagate(False)
        
        left_canvas = tk.Canvas(left_panel_container, bg='white', highlightthickness=0)
        left_scrollbar = tk.Scrollbar(left_panel_container, orient='vertical', command=left_canvas.yview)
        left_panel = tk.Frame(left_canvas, bg='white')
        
        left_panel.bind(
            '<Configure>',
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox('all'))
        )
        
        left_canvas.create_window((0, 0), window=left_panel, anchor='nw')
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side='left', fill='both', expand=True)
        left_scrollbar.pack(side='right', fill='y')
        
        def _on_mouse_wheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_mousewheel(event):
            left_canvas.bind_all("<MouseWheel>", _on_mouse_wheel)
        
        def _unbind_mousewheel(event):
            left_canvas.unbind_all("<MouseWheel>")
        
        left_panel_container.bind('<Enter>', _bind_mousewheel)
        left_panel_container.bind('<Leave>', _unbind_mousewheel)
        
        self.create_control_panel(left_panel)
        
        # Right Panel
        viz_frame = tk.LabelFrame(
            main_container,
            text="📊 Test Data Anomaly Visualization",
            font=('Arial', 13, 'bold'),
            bg='white',
            fg='#e74c3c',
            relief='raised',
            borderwidth=2
        )
        viz_frame.pack(side='left', fill='both', expand=True)
        
        scroll_frame = tk.Frame(viz_frame, bg='white')
        scroll_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = tk.Scrollbar(scroll_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        self.canvas_widget = tk.Canvas(
            scroll_frame,
            bg='white',
            yscrollcommand=scrollbar.set,
            highlightthickness=0
        )
        self.canvas_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.canvas_widget.yview)
        
        self.canvas = tk.Frame(self.canvas_widget, bg='white')
        self.canvas_window = self.canvas_widget.create_window(0, 0, window=self.canvas, anchor='nw')
        
        self.canvas.bind('<Configure>', lambda e: self.canvas_widget.configure(scrollregion=self.canvas_widget.bbox('all')))
        self.canvas_widget.bind('<Configure>', lambda e: self.canvas_widget.itemconfig(self.canvas_window, width=e.width))
        
        def _on_viz_wheel(event):
            self.canvas_widget.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_viz_wheel(event):
            self.canvas_widget.bind_all("<MouseWheel>", _on_viz_wheel)
        
        def _unbind_viz_wheel(event):
            self.canvas_widget.unbind_all("<MouseWheel>")
        
        viz_frame.bind('<Enter>', _bind_viz_wheel)
        viz_frame.bind('<Leave>', _unbind_viz_wheel)
        
        self.show_welcome()
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text=self.get_status(),
            bd=1,
            relief='sunken',
            anchor='w',
            bg='#bdc3c7',
            font=('Arial', 10)
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def show_welcome(self):
        """Show welcome message"""
        tk.Label(
            self.canvas,
            text="📁 Load Test Data to Begin\n\n"
                 "Select features and visualization mode\n"
                 "Anomalies will trigger SMS alerts automatically",
            font=('Arial', 12),
            bg='white',
            fg='#95a5a6',
            justify='center'
        ).pack(expand=True, pady=50)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        tk.Label(
            parent,
            text="Control Panel",
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=10)
        
        # Alert Settings
        alert_frame = tk.LabelFrame(parent, text="📱 SMS Alert Settings", font=('Arial', 10, 'bold'), 
                                     bg='white', padx=8, pady=6)
        alert_frame.pack(fill='x', padx=10, pady=8)
        
        # Twilio status
        status_color = '#27ae60' if self.twilio_configured else '#e74c3c'
        status_text = '✓ Connected' if self.twilio_configured else '✗ Not Configured'
        tk.Label(alert_frame, text=f"Twilio: {status_text}", 
                bg='white', fg=status_color, font=('Arial', 9, 'bold')).pack(pady=5)
        
        self.send_sms = tk.BooleanVar(value=True)
        tk.Checkbutton(alert_frame, text="Send SMS Alerts", variable=self.send_sms,
                      bg='white', font=('Arial', 9, 'bold')).pack(anchor='w', padx=5)
        
        tk.Label(alert_frame, text="Max Alerts per Run:", bg='white', 
                font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8,2), padx=5)
        self.max_alerts = tk.Spinbox(alert_frame, from_=1, to=50, width=12,
                                     font=('Arial', 10))
        self.max_alerts.delete(0, 'end')
        self.max_alerts.insert(0, '5')
        self.max_alerts.pack(anchor='w', padx=5)
        
        # Data Loading
        data_frame = tk.LabelFrame(parent, text="📁 Data", font=('Arial', 10, 'bold'), 
                                   bg='white', padx=8, pady=6)
        data_frame.pack(fill='x', padx=10, pady=8)
        
        self.train_label = tk.Label(data_frame, 
                                     text="Training: " + ("✓" if self.training_df is not None else "✗"),
                                     bg='white', fg='#27ae60' if self.training_df is not None else '#e74c3c',
                                     font=('Arial', 9))
        self.train_label.pack(pady=2)
        
        tk.Button(
            data_frame,
            text="📊 Load Test Data",
            command=self.load_test_data,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            padx=8,
            pady=5
        ).pack(fill='x', pady=3)
        
        self.test_label = tk.Label(data_frame, text="Not loaded", bg='white', fg='#95a5a6', font=('Arial', 8))
        self.test_label.pack(pady=2)
        
        # Visualization Mode
        viz_frame = tk.LabelFrame(parent, text="🎯 Visualization Mode", 
                                  font=('Arial', 10, 'bold'), bg='white', padx=8, pady=6)
        viz_frame.pack(fill='x', padx=10, pady=8)
        
        self.viz_mode = tk.StringVar(value="multi")
        
        tk.Radiobutton(
            viz_frame,
            text="Multi-Feature Timeline",
            variable=self.viz_mode,
            value="multi",
            bg='white',
            font=('Arial', 9)
        ).pack(anchor='w', pady=2)
        
        tk.Radiobutton(
            viz_frame,
            text="Single Feature Detailed",
            variable=self.viz_mode,
            value="single",
            bg='white',
            font=('Arial', 9)
        ).pack(anchor='w', pady=2)
        
        # Feature Selection
        feature_frame = tk.LabelFrame(parent, text="📋 Feature Selection", 
                                      font=('Arial', 10, 'bold'), bg='white', padx=8, pady=6)
        feature_frame.pack(fill='x', padx=10, pady=8)
        
        tk.Label(feature_frame, text="Feature Group:", bg='white', 
                font=('Arial', 9, 'bold')).pack(anchor='w', pady=(3, 1))
        
        self.group_combo = ttk.Combobox(
            feature_frame,
            values=list(self.feature_groups.keys()),
            state='readonly',
            font=('Arial', 9),
            width=28
        )
        self.group_combo.set('Energy Consumption')
        self.group_combo.pack(fill='x', pady=2)
        self.group_combo.bind('<<ComboboxSelected>>', self.on_group_change)
        
        tk.Label(feature_frame, text="Specific Feature:", bg='white', 
                font=('Arial', 9, 'bold')).pack(anchor='w', pady=(8, 1))
        
        self.feature_combo = ttk.Combobox(
            feature_frame,
            values=self.feature_groups['Energy Consumption'],
            state='readonly',
            font=('Arial', 9),
            width=28
        )
        self.feature_combo.set('Car charger [kW]')
        self.feature_combo.pack(fill='x', pady=2)
        
        # Visualize Button
        tk.Button(
            feature_frame,
            text="🚨 Detect & Send Alerts",
            command=self.visualize_anomalies,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            cursor='hand2',
            padx=10,
            pady=8
        ).pack(fill='x', pady=(10, 3))
        
        # Actions
        action_frame = tk.Frame(parent, bg='white')
        action_frame.pack(fill='x', padx=10, pady=8)
        
        tk.Button(
            action_frame,
            text="🗑️ Clear",
            command=self.clear_viz,
            bg='#95a5a6',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            padx=6,
            pady=5
        ).pack(side='left', fill='x', expand=True, padx=2)
        
        tk.Button(
            action_frame,
            text="❓ Help",
            command=self.show_help,
            bg='#16a085',
            fg='white',
            font=('Arial', 9, 'bold'),
            cursor='hand2',
            padx=6,
            pady=5
        ).pack(side='right', fill='x', expand=True, padx=2)
    
    def on_group_change(self, event=None):
        """Update feature list when group changes"""
        group = self.group_combo.get()
        features = self.feature_groups.get(group, [])
        self.feature_combo['values'] = features
        if features:
            self.feature_combo.set(features[0])
    
    def load_test_data(self):
        """Load test dataset"""
        file_path = filedialog.askopenfilename(
            title="Select Test Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.test_df = self.data_processor.load_dataset(file_path)
                if self.test_df is not None:
                    self.test_label.config(
                        text=f"✓ {os.path.basename(file_path)[:20]}\n{len(self.test_df)} records",
                        fg='#27ae60'
                    )
                    self.update_status("Test data loaded")
                    messagebox.showinfo("Success", f"Test data loaded!\n{len(self.test_df)} records")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load:\n{str(e)}")
    
    def visualize_anomalies(self):
        """Visualize anomalies and send alerts"""
        if self.test_df is None:
            messagebox.showwarning("Warning", "Please load test data first!")
            return
        
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded! Train models first.")
            return
        
        mode = self.viz_mode.get()
        group = self.group_combo.get()
        feature = self.feature_combo.get()
        
        self.update_status(f"Detecting anomalies...")
        
        try:
            if mode == "multi":
                features = self.feature_groups.get(group, [])
                result = self.visualizer.plot_multi_feature_timeline(
                    self.test_df, features, self.canvas, self.data_processor
                )
            else:
                result = self.visualizer.plot_single_feature_detailed(
                    self.test_df, feature, self.canvas, self.data_processor
                )
            
            if result:
                anomaly_count = len(result['anomaly_indices'])
                anomaly_pct = (anomaly_count / len(self.test_df)) * 100
                
                print("\n" + "="*70)
                print(f"⚠️  ANOMALY DETECTION COMPLETE")
                print("="*70)
                print(f"Total Anomalies Found: {anomaly_count} ({anomaly_pct:.2f}%)")
                print("-"*70)
                
                # Send alerts if configured
                if self.twilio_configured and self.send_sms.get() and anomaly_count > 0:
                    max_alerts = int(self.max_alerts.get())
                    
                    print(f"\n📱 Sending SMS alerts (max: {max_alerts})...")
                    print("="*70)
                    
                    # Get top features for alert
                    alert_features = [col for col in self.visualizer.feature_columns 
                                     if col in self.test_df.columns][:5]
                    
                    # Send alerts for each anomaly
                    for i, idx in enumerate(result['anomaly_indices'][:max_alerts]):
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Collect sensor data
                        sensor_data = {}
                        for col in alert_features:
                            if col in self.test_df.columns:
                                value = self.test_df.iloc[idx][col]
                                sensor_data[col] = value
                        
                        # Send alert
                        self.alert_system.send_alert(
                            timestamp=timestamp,
                            sensor_data=sensor_data,
                            prediction=-1,
                            include_call=False
                        )
                    
                    if anomaly_count > max_alerts:
                        print(f"\n⚠️  Note: {anomaly_count - max_alerts} additional anomalies not alerted (limit: {max_alerts})")
                    
                    print("="*70 + "\n")
                
                self.update_status(f"✓ Anomalies detected: {anomaly_count} ({anomaly_pct:.2f}%)")
                
        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed:\n{str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_viz(self):
        """Clear visualization"""
        for widget in self.canvas.winfo_children():
            widget.destroy()
        self.show_welcome()
        self.canvas_widget.yview_moveto(0)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
IoT ANOMALY DETECTION WITH SMS ALERTS
═════════════════════════════════════

🎯 WORKFLOW:

1️⃣ Load Test Data:
   • Upload CSV file to analyze
   • System detects anomalies automatically

2️⃣ Configure Alert Settings:
   • Enable/disable SMS alerts
   • Set max alerts to avoid spam (default: 5)

3️⃣ Select Visualization:
   • Multi-Feature Timeline: View multiple sensors
   • Single Feature Detailed: Deep dive into one sensor

4️⃣ Detect & Send Alerts:
   • Click "Detect & Send Alerts"
   • Anomalies trigger real-time SMS
   • View results in visualization panel

═════════════════════════════════════

📱 TERMINAL OUTPUT FORMAT:

2025-10-07 15:23:12 | {Temperature=38.9, Motion=1, ...} | pred=-1
⚠️ Anomaly detected at 2025-10-07 15:23:12! Temperature=38.9, Motion=1
📩 SMS sent: SID=SMxxxxxxxxxxxxxxxx

═════════════════════════════════════

💡 TIPS:

• Twilio credentials are pre-configured in twilio_alert.py
• Limit max alerts to avoid SMS spam
• Check terminal for detailed alert logs
• All anomalies are visualized on the graph

═════════════════════════════════════

⚙️ CONFIGURATION:

Edit twilio_alert.py to change:
• Account SID
• Auth Token  
• From/To phone numbers

═════════════════════════════════════
        """
        messagebox.showinfo("Help", help_text)
    
    def get_status(self):
        """Get status text"""
        status = "Ready"
        if self.training_df is not None:
            status += " | Training: ✓"
        if self.test_df is not None:
            status += " | Test: ✓"
        if self.models_loaded:
            status += " | Models: ✓"
        if self.twilio_configured:
            status += " | Twilio: ✓"
        return status
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=f"{message} | {self.get_status()}")
        self.root.update_idletasks()


def main():
    """Main function"""
    print("="*70)
    print("🏠 IoT Smart Home Anomaly Detection with Twilio SMS Alerts")
    print("="*70)
    print("Credentials loaded from twilio_alert.py")
    print("Anomalies will trigger SMS alerts and terminal logs")
    print("="*70 + "\n")
    
    root = tk.Tk()
    app = AnomalyDetectionApp(root)
    
    print("✓ Application started successfully!")
    print("• Load test data to begin analysis")
    print("• Anomalies trigger real-time SMS notifications")
    print("• Check terminal for detailed alert logs\n")
    
    root.mainloop()


if __name__ == "__main__":
    main()