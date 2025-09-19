import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingEngine:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ARIMA': None  # Placeholder for ARIMA implementation
        }
        self.scaler = StandardScaler()
    
    def generate_forecast(self, product_data, forecast_days, model_type='Random Forest'):
        """Generate forecast for a product"""
        try:
            # Prepare data
            prepared_data = self._prepare_data(product_data)
            
            if prepared_data is None or len(prepared_data) < 30:
                logger.warning(f"Insufficient data for forecasting")
                return None
            
            # Create features and target
            X, y = self._create_features(prepared_data)
            
            if X is None or len(X) == 0:
                logger.warning("Could not create features for forecasting")
                return None
            
            # Split data for validation
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Train model
            model = self._get_model(model_type)
            if model is None:
                return None
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate accuracy metrics
            mape = self._calculate_mape(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            
            # Generate future predictions
            future_features = self._create_future_features(prepared_data, forecast_days)
            future_features_scaled = self.scaler.transform(future_features)
            future_predictions = model.predict(future_features_scaled)
            
            # Create forecast dataframe
            last_date = prepared_data['date'].max()
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'predicted_quantity': np.maximum(future_predictions, 0)  # Ensure non-negative predictions
            })
            
            # Add confidence intervals (simplified approach)
            std_error = np.std(y_test - y_pred_test)
            forecast_df['confidence_lower'] = np.maximum(forecast_df['predicted_quantity'] - 1.96 * std_error, 0)
            forecast_df['confidence_upper'] = forecast_df['predicted_quantity'] + 1.96 * std_error
            
            # Prepare historical data for visualization
            historical_df = prepared_data[['date', 'sales_quantity']].copy()
            
            result = {
                'historical': historical_df,
                'forecast': forecast_df,
                'model_type': model_type,
                'accuracy': 100 - mape,
                'mape': mape,
                'rmse': rmse,
                'mae': mae
            }
            
            logger.info(f"Forecast generated successfully with MAPE: {mape:.2f}%")
            return result
        
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return None
    
    def _prepare_data(self, data):
        """Prepare and clean data for forecasting"""
        try:
            df = data.copy()
            
            # Ensure date column is datetime
            if 'date' not in df.columns:
                logger.error("Date column not found in data")
                return None
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Ensure sales_quantity column exists
            if 'sales_quantity' not in df.columns:
                logger.error("Sales quantity column not found in data")
                return None
            
            # Sort by date
            df = df.sort_values('date')
            
            # Aggregate by date (sum sales for same dates)
            df = df.groupby('date').agg({
                'sales_quantity': 'sum',
                'revenue': 'sum' if 'revenue' in df.columns else lambda x: 0
            }).reset_index()
            
            # Fill missing dates
            date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            df = df.set_index('date').reindex(date_range, fill_value=0).reset_index()
            df.columns = ['date', 'sales_quantity', 'revenue'] if 'revenue' in df.columns else ['date', 'sales_quantity']
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None
    
    def _create_features(self, data):
        """Create features for machine learning"""
        try:
            df = data.copy()
            
            # Create time-based features
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Create lag features
            for lag in [1, 7, 14, 30]:
                df[f'lag_{lag}'] = df['sales_quantity'].shift(lag)
            
            # Create rolling statistics
            for window in [7, 14, 30]:
                df[f'rolling_mean_{window}'] = df['sales_quantity'].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df['sales_quantity'].rolling(window=window, min_periods=1).std().fillna(0)
            
            # Create trend feature
            df['trend'] = range(len(df))
            
            # Drop rows with NaN values in lag features
            feature_columns = [col for col in df.columns if col not in ['date', 'sales_quantity', 'revenue']]
            df = df.dropna(subset=feature_columns)
            
            if len(df) < 10:
                logger.warning("Insufficient data after feature engineering")
                return None, None
            
            X = df[feature_columns].values
            y = df['sales_quantity'].values
            
            return X, y
        
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return None, None
    
    def _create_future_features(self, data, forecast_days):
        """Create features for future predictions"""
        try:
            df = data.copy()
            last_date = df['date'].max()
            
            # Create future dates
            future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
            
            future_features = []
            
            for i, future_date in enumerate(future_dates):
                features = {}
                
                # Time-based features
                features['day_of_week'] = future_date.weekday()
                features['day_of_month'] = future_date.day
                features['month'] = future_date.month
                features['quarter'] = (future_date.month - 1) // 3 + 1
                features['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
                
                # For lag features, we need to use recent actual data and predictions
                current_data = df['sales_quantity'].tolist()
                
                # Add previous predictions to current_data for lag calculation
                if i > 0:
                    for j in range(i):
                        current_data.append(future_features[j].get('predicted_quantity', current_data[-1]))
                
                # Calculate lag features
                for lag in [1, 7, 14, 30]:
                    if len(current_data) >= lag:
                        features[f'lag_{lag}'] = current_data[-lag]
                    else:
                        features[f'lag_{lag}'] = current_data[-1] if current_data else 0
                
                # Calculate rolling statistics using recent data
                for window in [7, 14, 30]:
                    recent_data = current_data[-window:] if len(current_data) >= window else current_data
                    if recent_data:
                        features[f'rolling_mean_{window}'] = np.mean(recent_data)
                        features[f'rolling_std_{window}'] = np.std(recent_data) if len(recent_data) > 1 else 0
                    else:
                        features[f'rolling_mean_{window}'] = 0
                        features[f'rolling_std_{window}'] = 0
                
                # Trend feature
                features['trend'] = len(df) + i
                
                future_features.append(features)
            
            # Convert to array format matching training features
            feature_columns = [col for col in df.columns if col not in ['date', 'sales_quantity', 'revenue']]
            future_array = []
            
            for features in future_features:
                row = [features.get(col, 0) for col in feature_columns]
                future_array.append(row)
            
            return np.array(future_array)
        
        except Exception as e:
            logger.error(f"Error creating future features: {e}")
            return np.array([])
    
    def _get_model(self, model_type):
        """Get model instance"""
        if model_type == 'ARIMA':
            # Simple moving average as ARIMA substitute
            return SimpleMovingAverageModel()
        elif model_type in self.models:
            return self.models[model_type]
        else:
            logger.warning(f"Unknown model type: {model_type}, using Random Forest")
            return self.models['Random Forest']
    
    def _calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        try:
            actual, predicted = np.array(actual), np.array(predicted)
            # Avoid division by zero
            mask = actual != 0
            if mask.sum() == 0:
                return 0
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        except:
            return 0

class SimpleMovingAverageModel:
    """Simple moving average model as ARIMA substitute"""
    
    def __init__(self, window=7):
        self.window = window
        self.data = None
    
    def fit(self, X, y):
        """Fit the model (store the target values)"""
        self.data = y
        return self
    
    def predict(self, X):
        """Predict using moving average"""
        if self.data is None:
            return np.zeros(len(X))
        
        # Use the last 'window' values to predict
        recent_values = self.data[-self.window:]
        prediction = np.mean(recent_values) if len(recent_values) > 0 else 0
        
        return np.full(len(X), prediction)
