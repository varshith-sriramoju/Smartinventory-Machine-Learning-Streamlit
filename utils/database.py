import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import streamlit as st
from datetime import datetime, timedelta
import logging
import numpy as np

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
                logger.info("Database initialized successfully")
            else:
                logger.warning("No database connection string available. Using in-memory storage.")
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
        logger.info("Using in-memory storage")
    
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
            year INTEGER,
            month INTEGER,
            day_of_week INTEGER,
            is_weekend INTEGER,
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
            forecasted_demand DECIMAL(10,2),
            safety_stock DECIMAL(10,2),
            reorder_point DECIMAL(10,2),
            eoq DECIMAL(10,2),
            recommended_order_quantity DECIMAL(10,2),
            action VARCHAR(50),
            urgency_score DECIMAL(5,2),
            excess_inventory DECIMAL(10,2),
            service_level DECIMAL(5,2),
            lead_time INTEGER,
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
        CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory_recommendations(product_name);
        CREATE INDEX IF NOT EXISTS idx_performance_product ON model_performance(product_name);
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
                df.to_sql('sales_data', self.engine, if_exists='replace', index=False, method='multi')
                logger.info(f"Saved {len(df)} sales records to database")
                return True
            else:
                # Fallback to session state
                st.session_state.sales_data = df
                logger.info(f"Saved {len(df)} sales records to session state")
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
                
                df = pd.read_sql(query, self.engine, params=params)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                # Fallback to session state
                df = st.session_state.sales_data.copy()
                
                if df.empty:
                    return df
                
                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                if product_name and not df.empty:
                    df = df[df['product_name'] == product_name]
                
                if start_date and not df.empty:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                
                if end_date and not df.empty:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                
                return df.sort_values('date') if not df.empty else df
        
        except Exception as e:
            logger.error(f"Error retrieving sales data: {e}")
            return pd.DataFrame()
    
    def get_total_records(self):
        """Get total number of records in database"""
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT COUNT(*) FROM sales_data"))
                    return result.scalar()
            else:
                return len(st.session_state.sales_data) if hasattr(st.session_state, 'sales_data') else 0
        except Exception as e:
            logger.error(f"Error getting total records: {e}")
            return 0
    
    def get_sample_data(self, limit=100):
        """Get sample data from database"""
        try:
            if self.engine:
                query = f"SELECT * FROM sales_data ORDER BY date DESC LIMIT {limit}"
                return pd.read_sql(query, self.engine)
            else:
                df = st.session_state.sales_data
                return df.head(limit) if not df.empty else df
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return pd.DataFrame()
    
    def clear_all_data(self):
        """Clear all data from database"""
        try:
            if self.engine:
                with self.engine.connect() as conn:
                    conn.execute(text("TRUNCATE TABLE sales_data, forecasts, inventory_recommendations, model_performance RESTART IDENTITY"))
                    conn.commit()
                logger.info("All database tables cleared")
            else:
                # Clear session state
                st.session_state.sales_data = pd.DataFrame()
                st.session_state.forecasts = pd.DataFrame()
                st.session_state.inventory_recommendations = pd.DataFrame()
                st.session_state.model_performance = pd.DataFrame()
                logger.info("All session state data cleared")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
    
    def save_forecast(self, product_name, forecast_data):
        """Save forecast data to database"""
        try:
            forecast_df = forecast_data['forecast'].copy()
            forecast_df['product_name'] = product_name
            forecast_df['model_type'] = forecast_data.get('model_type', 'unknown')
            forecast_df['created_at'] = datetime.now()
            
            # Rename columns to match database schema
            forecast_df = forecast_df.rename(columns={
                'date': 'forecast_date'
            })
            
            # Ensure required columns exist
            required_cols = ['product_name', 'forecast_date', 'predicted_quantity', 'model_type', 'created_at']
            for col in required_cols:
                if col not in forecast_df.columns:
                    if col == 'confidence_lower':
                        forecast_df[col] = None
                    elif col == 'confidence_upper':
                        forecast_df[col] = None
            
            if self.engine:
                forecast_df.to_sql('forecasts', self.engine, if_exists='append', index=False)
                logger.info(f"Saved forecast for {product_name}")
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
                df = pd.read_sql(query, self.engine, params={'limit': limit})
                if not df.empty:
                    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
                    df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                # Fallback to session state
                df = st.session_state.forecasts
                if not df.empty and 'created_at' in df.columns:
                    df = df.sort_values('created_at', ascending=False).head(limit)
                return df
        
        except Exception as e:
            logger.error(f"Error retrieving forecasts: {e}")
            return pd.DataFrame()
    
    def save_inventory_recommendations(self, recommendations_df):
        """Save inventory recommendations to database"""
        try:
            recommendations_df = recommendations_df.copy()
            recommendations_df['created_at'] = datetime.now()
            
            if self.engine:
                recommendations_df.to_sql('inventory_recommendations', self.engine, if_exists='replace', index=False)
                logger.info(f"Saved {len(recommendations_df)} inventory recommendations")
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
            
            # Calculate average sales and simulate current inventory
            product_stats = sales_data.groupby('product_name')['sales_quantity'].agg(['mean', 'std']).reset_index()
            product_stats['std'] = product_stats['std'].fillna(0)
            
            # Simulate current inventory as 15-45 days of average sales plus some randomness
            np.random.seed(42)  # For consistent results
            product_stats['current_stock'] = (
                product_stats['mean'] * np.random.uniform(15, 45, len(product_stats)) +
                product_stats['std'] * np.random.uniform(0, 10, len(product_stats))
            )
            product_stats['current_stock'] = product_stats['current_stock'].fillna(50).clip(lower=5)
            
            return product_stats[['product_name', 'current_stock']]
        
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
            
            # Calculate recent performance
            recent_data = sales_data[sales_data['date'] >= (sales_data['date'].max() - timedelta(days=30))]
            older_data = sales_data[
                (sales_data['date'] >= (sales_data['date'].max() - timedelta(days=60))) &
                (sales_data['date'] < (sales_data['date'].max() - timedelta(days=30)))
            ]
            
            recent_revenue = recent_data['revenue'].sum() if 'revenue' in recent_data.columns else 0
            older_revenue = older_data['revenue'].sum() if 'revenue' in older_data.columns else 1
            revenue_growth = ((recent_revenue - older_revenue) / max(older_revenue, 1)) * 100
            
            # Get forecast accuracy from model performance
            performance_data = self.get_model_performance()
            avg_accuracy = 85.0  # Default
            accuracy_trend = 2.1  # Default
            
            if not performance_data.empty:
                avg_accuracy = 100 - performance_data['mape'].mean()
                if len(performance_data) > 10:
                    recent_accuracy = 100 - performance_data.tail(5)['mape'].mean()
                    older_accuracy = 100 - performance_data.head(5)['mape'].mean()
                    accuracy_trend = recent_accuracy - older_accuracy
            
            # Simulate some additional metrics
            low_stock_items = max(5, int(active_products * 0.1))  # 10% of products
            
            summary = {
                'total_revenue': total_revenue,
                'revenue_growth': revenue_growth,
                'active_products': active_products,
                'new_products': max(1, int(active_products * 0.05)),  # 5% new products
                'forecast_accuracy': avg_accuracy,
                'accuracy_trend': accuracy_trend,
                'low_stock_items': low_stock_items,
                'stock_trend': np.random.randint(-5, 3)  # Random stock trend
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"Error getting summary statistics: {e}")
            return None
    
    def get_inventory_alerts(self):
        """Get inventory alerts (low stock, overstock)"""
        try:
            # Get current inventory and recent sales to calculate alerts
            current_inventory = self.get_current_inventory()
            sales_data = self.get_sales_data()
            
            if current_inventory.empty or sales_data.empty:
                return pd.DataFrame()
            
            # Calculate average daily sales for each product
            daily_sales = sales_data.groupby('product_name')['sales_quantity'].mean()
            
            alerts = []
            for _, row in current_inventory.iterrows():
                product = row['product_name']
                current_stock = row['current_stock']
                
                if product in daily_sales:
                    avg_daily_sales = daily_sales[product]
                    days_of_stock = current_stock / max(avg_daily_sales, 0.1)
                    
                    if days_of_stock < 7:  # Less than 1 week of stock
                        alerts.append({
                            'product_name': product,
                            'current_stock': current_stock,
                            'alert_type': 'low_stock'
                        })
                    elif days_of_stock > 90:  # More than 3 months of stock
                        alerts.append({
                            'product_name': product,
                            'current_stock': current_stock,
                            'alert_type': 'overstock'
                        })
            
            return pd.DataFrame(alerts)
        
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
                logger.info(f"Saved performance metrics for {product_name}")
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
                df = pd.read_sql(query, self.engine)
                if not df.empty:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                return df
            else:
                # Fallback to session state
                return st.session_state.model_performance if hasattr(st.session_state, 'model_performance') else pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return pd.DataFrame()
