import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import streamlit as st
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.engine = None
        self._initialize_database()
    
    def _get_connection_string(self):
        """Get database connection string from environment variables"""
        try:
            # Try to get full DATABASE_URL first
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                return db_url
            
            # Fall back to individual components
            host = os.getenv('PGHOST', 'localhost')
            port = os.getenv('PGPORT', '5432')
            database = os.getenv('PGDATABASE', 'smartinventory')
            user = os.getenv('PGUSER', 'postgres')
            password = os.getenv('PGPASSWORD', '')
            
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        except Exception as e:
            logger.error(f"Error getting connection string: {e}")
            return None
    
    def _initialize_database(self):
        """Initialize database connection and create tables if they don't exist"""
        try:
            if self.connection_string:
                self.engine = create_engine(self.connection_string)
                self._create_tables()
            else:
                logger.warning("No database connection string available. Using in-memory storage.")
                # Initialize in-memory storage as fallback
                self._init_memory_storage()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._init_memory_storage()
    
    def _init_memory_storage(self):
        """Initialize in-memory storage as fallback"""
        if 'sales_data' not in st.session_state:
            st.session_state.sales_data = pd.DataFrame()
        if 'forecasts' not in st.session_state:
            st.session_state.forecasts = pd.DataFrame()
        if 'inventory_recommendations' not in st.session_state:
            st.session_state.inventory_recommendations = pd.DataFrame()
        if 'model_performance' not in st.session_state:
            st.session_state.model_performance = pd.DataFrame()
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        if not self.engine:
            return
        
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS sales_data (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            product_id VARCHAR(100),
            product_name VARCHAR(255) NOT NULL,
            category VARCHAR(100),
            store_id VARCHAR(100),
            sales_quantity DECIMAL(10,2) NOT NULL,
            price DECIMAL(10,2),
            revenue DECIMAL(12,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS forecasts (
            id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_quantity DECIMAL(10,2) NOT NULL,
            confidence_lower DECIMAL(10,2),
            confidence_upper DECIMAL(10,2),
            model_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS inventory_recommendations (
            id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            current_stock DECIMAL(10,2),
            safety_stock DECIMAL(10,2),
            recommended_order_quantity DECIMAL(10,2),
            action VARCHAR(50),
            urgency_score DECIMAL(5,2),
            excess_inventory DECIMAL(10,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            mape DECIMAL(8,4),
            rmse DECIMAL(10,4),
            mae DECIMAL(10,4),
            forecast_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_sales_date ON sales_data(date);
        CREATE INDEX IF NOT EXISTS idx_sales_product ON sales_data(product_name);
        CREATE INDEX IF NOT EXISTS idx_forecasts_product_date ON forecasts(product_name, forecast_date);
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_tables_sql))
                conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def save_sales_data(self, df):
        """Save sales data to database"""
        try:
            if self.engine:
                df.to_sql('sales_data', self.engine, if_exists='replace', index=False)
                return True
            else:
                # Fallback to session state
                st.session_state.sales_data = df
                return True
        except Exception as e:
            logger.error(f"Error saving sales data: {e}")
            return False
    
    def get_sales_data(self, product_name=None, start_date=None, end_date=None):
        """Retrieve sales data from database"""
        try:
            if self.engine:
                query = "SELECT * FROM sales_data WHERE 1=1"
                params = {}
                
                if product_name:
                    query += " AND product_name = %(product_name)s"
                    params['product_name'] = product_name
                
                if start_date:
                    query += " AND date >= %(start_date)s"
                    params['start_date'] = start_date
                
                if end_date:
                    query += " AND date <= %(end_date)s"
                    params['end_date'] = end_date
                
                query += " ORDER BY date"
                
                return pd.read_sql(query, self.engine, params=params)
            else:
                # Fallback to session state
                df = st.session_state.sales_data.copy()
                
                if product_name and not df.empty:
                    df = df[df['product_name'] == product_name]
                
                if start_date and not df.empty:
                    df = df[df['date'] >= start_date]
                
                if end_date and not df.empty:
                    df = df[df['date'] <= end_date]
                
                return df.sort_values('date') if not df.empty else df
        
        except Exception as e:
            logger.error(f"Error retrieving sales data: {e}")
            return pd.DataFrame()
    
    def save_forecast(self, product_name, forecast_data):
        """Save forecast data to database"""
        try:
            forecast_df = forecast_data['forecast'].copy()
            forecast_df['product_name'] = product_name
            forecast_df['model_type'] = forecast_data.get('model_type', 'unknown')
            forecast_df['created_at'] = datetime.now()
            
            # Rename columns to match database schema
            forecast_df = forecast_df.rename(columns={
                'date': 'forecast_date',
                'predicted_quantity': 'predicted_quantity'
            })
            
            if self.engine:
                forecast_df.to_sql('forecasts', self.engine, if_exists='append', index=False)
                return True
            else:
                # Fallback to session state
                if st.session_state.forecasts.empty:
                    st.session_state.forecasts = forecast_df
                else:
                    st.session_state.forecasts = pd.concat([st.session_state.forecasts, forecast_df], ignore_index=True)
                return True
        
        except Exception as e:
            logger.error(f"Error saving forecast: {e}")
            return False
    
    def get_recent_forecasts(self, limit=100):
        """Get recent forecasts from database"""
        try:
            if self.engine:
                query = """
                SELECT product_name, forecast_date, predicted_quantity, model_type, created_at
                FROM forecasts 
                ORDER BY created_at DESC 
                LIMIT %(limit)s
                """
                return pd.read_sql(query, self.engine, params={'limit': limit})
            else:
                # Fallback to session state
                df = st.session_state.forecasts
                if not df.empty:
                    return df.sort_values('created_at', ascending=False).head(limit)
                return df
        
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return pd.DataFrame()
    
    def save_inventory_recommendations(self, recommendations_df):
        """Save inventory recommendations to database"""
        try:
            recommendations_df['created_at'] = datetime.now()
            
            if self.engine:
                recommendations_df.to_sql('inventory_recommendations', self.engine, if_exists='replace', index=False)
                return True
            else:
                # Fallback to session state
                st.session_state.inventory_recommendations = recommendations_df
                return True
        
        except Exception as e:
            logger.error(f"Error saving inventory recommendations: {e}")
            return False
    
    def get_current_inventory(self):
        """Get current inventory levels (simulated data for demo)"""
        try:
            # In a real system, this would query an inventory management system
            # For demo purposes, we'll simulate current inventory based on recent sales
            sales_data = self.get_sales_data()
            
            if sales_data.empty:
                return pd.DataFrame()
            
            # Simulate current inventory as random values based on recent sales patterns
            recent_sales = sales_data.groupby('product_name')['sales_quantity'].agg(['mean', 'std']).reset_index()
            recent_sales['current_stock'] = recent_sales['mean'] * 30 + (recent_sales['std'] * 5)  # Simulate 30 days of average sales plus buffer
            recent_sales['current_stock'] = recent_sales['current_stock'].fillna(100).clip(lower=0)
            
            return recent_sales[['product_name', 'current_stock']]
        
        except Exception as e:
            logger.error(f"Error getting current inventory: {e}")
            return pd.DataFrame()
    
    def get_summary_statistics(self):
        """Get summary statistics for dashboard"""
        try:
            sales_data = self.get_sales_data()
            
            if sales_data.empty:
                return None
            
            # Calculate summary statistics
            total_revenue = sales_data['revenue'].sum() if 'revenue' in sales_data.columns else 0
            active_products = sales_data['product_name'].nunique()
            
            # Simulate some metrics for demo
            summary = {
                'total_revenue': total_revenue,
                'revenue_growth': 5.2,  # Simulated
                'active_products': active_products,
                'new_products': 3,  # Simulated
                'forecast_accuracy': 87.5,  # Simulated
                'accuracy_trend': 2.1,  # Simulated
                'low_stock_items': 12,  # Simulated
                'stock_trend': -2  # Simulated
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting summary statistics: {e}")
            return None
    
    def get_inventory_alerts(self):
        """Get inventory alerts (low stock, overstock)"""
        try:
            # This would normally query real inventory data
            # For demo, we'll return some simulated alerts
            alerts_data = [
                {'product_name': 'Product A', 'current_stock': 5, 'alert_type': 'low_stock'},
                {'product_name': 'Product B', 'current_stock': 2, 'alert_type': 'low_stock'},
                {'product_name': 'Product C', 'current_stock': 500, 'alert_type': 'overstock'}
            ]
            
            return pd.DataFrame(alerts_data) if alerts_data else pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error getting inventory alerts: {e}")
            return pd.DataFrame()
    
    def save_model_performance(self, product_name, model_type, mape, rmse, mae):
        """Save model performance metrics"""
        try:
            performance_data = {
                'product_name': product_name,
                'model_type': model_type,
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'created_at': datetime.now()
            }
            
            if self.engine:
                pd.DataFrame([performance_data]).to_sql('model_performance', self.engine, if_exists='append', index=False)
                return True
            else:
                # Fallback to session state
                if st.session_state.model_performance.empty:
                    st.session_state.model_performance = pd.DataFrame([performance_data])
                else:
                    st.session_state.model_performance = pd.concat([st.session_state.model_performance, pd.DataFrame([performance_data])], ignore_index=True)
                return True
        
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
            return False
    
    def get_model_performance(self):
        """Get model performance data"""
        try:
            if self.engine:
                query = "SELECT * FROM model_performance ORDER BY created_at DESC"
                return pd.read_sql(query, self.engine)
            else:
                # Fallback to session state
                return st.session_state.model_performance
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return pd.DataFrame()
