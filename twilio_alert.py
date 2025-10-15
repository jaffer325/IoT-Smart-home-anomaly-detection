"""
File: twilio_alert.py
Twilio SMS and Call Alerts for IoT Anomaly Detection
Sends SMS and initiates calls when anomalies are detected
"""

from twilio.rest import Client
from datetime import datetime
import pandas as pd

class TwilioAnomalyAlert:
    def __init__(self, account_sid, auth_token, from_phone, to_phone):
        """
        Initialize Twilio client
        
        Args:
            account_sid: Your Twilio Account SID
            auth_token: Your Twilio Auth Token
            from_phone: Your Twilio phone number (format: +1234567890)
            to_phone: Recipient phone number (format: +1234567890)
        """
        self.client = Client(account_sid, auth_token)
        self.from_phone = from_phone
        self.to_phone = to_phone
        
        print("="*60)
        print("Twilio Anomaly Alert System Initialized")
        print("="*60)
        print(f"From: {from_phone}")
        print(f"To: {to_phone}")
        print("="*60 + "\n")
    
    def send_sms_alert(self, timestamp, sensor_data, prediction=-1):
        """
        Send SMS alert for detected anomaly
        
        Args:
            timestamp: Timestamp of anomaly
            sensor_data: Dictionary of sensor readings
            prediction: Model prediction (-1 for anomaly)
        
        Returns:
            SMS SID if successful, None otherwise
        """
        try:
            # Format sensor data
            sensor_str = ", ".join([f"{k}={v}" for k, v in sensor_data.items()])
            
            # Create message
            message_body = (
                f"🚨 ANOMALY DETECTED!\n\n"
                f"Time: {timestamp}\n"
                f"Prediction: {prediction}\n\n"
                f"Sensor Data:\n{sensor_str}\n\n"
                f"⚠️ Please check your smart home system immediately!"
            )
            
            # Send SMS
            message = self.client.messages.create(
                body=message_body,
                from_=self.from_phone,
                to=self.to_phone
            )
            
            print(f"📩 SMS sent: SID={message.sid}")
            return message.sid
            
        except Exception as e:
            print(f"❌ SMS failed: {str(e)}")
            return None
    
    def make_alert_call(self, timestamp, sensor_data):
        """
        Initiate phone call for critical anomaly
        
        Args:
            timestamp: Timestamp of anomaly
            sensor_data: Dictionary of sensor readings
        
        Returns:
            Call SID if successful, None otherwise
        """
        try:
            # Create TwiML for voice message
            twiml_url = self._create_twiml_message(timestamp, sensor_data)
            
            # Make call
            call = self.client.calls.create(
                twiml=twiml_url,
                from_=self.from_phone,
                to=self.to_phone
            )
            
            print(f"📞 Call initiated: SID={call.sid}")
            return call.sid
            
        except Exception as e:
            print(f"❌ Call failed: {str(e)}")
            return None
    
    def _create_twiml_message(self, timestamp, sensor_data):
        """Create TwiML voice message"""
        # Format key sensor values for voice
        sensor_summary = []
        for key, value in list(sensor_data.items())[:4]:  # Top 4 sensors
            sensor_summary.append(f"{key} is {value}")
        
        sensor_text = ". ".join(sensor_summary)
        
        twiml = f"""
        <Response>
            <Say voice="alice">
                Warning! Anomaly detected in your smart home system at {timestamp}.
                {sensor_text}.
                Please check your system immediately.
            </Say>
            <Pause length="1"/>
            <Say voice="alice">
                This is an automated alert from your IoT anomaly detection system.
            </Say>
        </Response>
        """
        
        return twiml
    
    def send_alert(self, timestamp, sensor_data, prediction=-1, include_call=False):
        """
        Send complete alert (SMS and optionally call)
        
        Args:
            timestamp: Timestamp of anomaly
            sensor_data: Dictionary of sensor readings
            prediction: Model prediction
            include_call: Whether to also make a phone call
        
        Returns:
            Dictionary with SMS and Call SIDs
        """
        print("\n" + "="*60)
        print(f"⚠️ ANOMALY DETECTED at {timestamp}!")
        print("="*60)
        
        # Print sensor data
        sensor_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in sensor_data.items()])
        print(f"Sensor Data: {sensor_str}")
        print(f"Prediction: {prediction}")
        print("-"*60)
        
        # Send SMS
        sms_sid = self.send_sms_alert(timestamp, sensor_data, prediction)
        
        # Optionally make call
        call_sid = None
        if include_call:
            call_sid = self.make_alert_call(timestamp, sensor_data)
        
        print("="*60 + "\n")
        
        return {
            'sms_sid': sms_sid,
            'call_sid': call_sid,
            'timestamp': timestamp
        }
    
    def process_anomaly_dataframe(self, df, anomaly_indices, feature_columns, 
                                   send_call=False, max_alerts=5):
        """
        Process anomalies from dataframe and send alerts
        
        Args:
            df: Pandas DataFrame with sensor data
            anomaly_indices: Array of indices where anomalies detected
            feature_columns: List of feature column names to include
            send_call: Whether to make phone calls (default: False)
            max_alerts: Maximum number of alerts to send (to avoid spam)
        
        Returns:
            List of alert results
        """
        print(f"\n{'='*60}")
        print(f"Processing {len(anomaly_indices)} anomalies (max alerts: {max_alerts})")
        print(f"{'='*60}\n")
        
        alert_results = []
        
        # Limit alerts to avoid spam
        alerts_to_send = min(len(anomaly_indices), max_alerts)
        
        for i, idx in enumerate(anomaly_indices[:alerts_to_send]):
            # Get timestamp
            if 'timestamp' in df.columns:
                timestamp = df.iloc[idx]['timestamp']
            else:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get sensor data for this anomaly
            sensor_data = {}
            for col in feature_columns:
                if col in df.columns:
                    value = df.iloc[idx][col]
                    sensor_data[col] = value
            
            # Send alert
            result = self.send_alert(
                timestamp=timestamp,
                sensor_data=sensor_data,
                prediction=-1,
                include_call=send_call
            )
            
            alert_results.append(result)
        
        if len(anomaly_indices) > max_alerts:
            print(f"\n⚠️ Note: {len(anomaly_indices) - max_alerts} additional anomalies detected")
            print(f"   but not alerted (limit: {max_alerts} alerts per run)\n")
        
        return alert_results


# Configuration template
def get_twilio_config():
    """
    Return Twilio configuration dictionary
    Update these values with your Twilio credentials
    """
    config = {
        'account_sid': 'Replace with your Account SID',  # Replace with your Account SID
        'auth_token': ' Replace with your Auth Token',    # Replace with your Auth Token
        'from_phone': '+19405739121',         # Replace with your Twilio number
        'to_phone': '+918838790512'            # Replace with recipient number
    }
    return config


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Twilio Anomaly Alert System - Test Mode")
    print("="*60)
    
    # Get configuration
    config = get_twilio_config()
    
    # Check if configured
    if config['account_sid'] == 'YOUR_ACCOUNT_SID':
        print("\n⚠️ WARNING: Please configure your Twilio credentials first!")
        print("\nUpdate the following in twilio_alert.py:")
        print("1. account_sid - Your Twilio Account SID")
        print("2. auth_token - Your Twilio Auth Token")
        print("3. from_phone - Your Twilio phone number")
        print("4. to_phone - Recipient phone number")
        print("\nGet credentials from: https://console.twilio.com/")
        print("="*60 + "\n")
    else:
        # Initialize alert system
        alert_system = TwilioAnomalyAlert(
            account_sid=config['account_sid'],
            auth_token=config['auth_token'],
            from_phone=config['from_phone'],
            to_phone=config['to_phone']
        )
        
        # Test with sample anomaly
        print("Sending test anomaly alert...\n")
        
        sample_data = {
            'Temperature': 38.9,
            'Motion': 1,
            'Light': 120,
            'Sound': 0.92,
            'Humidity': 75.3
        }
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Send SMS alert (no call for test)
        result = alert_system.send_alert(
            timestamp=timestamp,
            sensor_data=sample_data,
            prediction=-1,
            include_call=False  # Set to True to test phone calls
        )
        
        print("Test completed!")
        print(f"Results: {result}")