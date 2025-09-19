import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.required_columns = ['date', 'product_name', 'sales_quantity']
        self.optional_columns = ['price', 'category', 'store_id']
    
    def validate_data(self, df):
        """Validate uploaded data"""
        try:
            errors = []
            
            # Check if dataframe is empty
            if df.empty:
                errors.append("File is empty")
                return {'is_valid': False, 'errors': errors}
            
            # Check minimum rows
            if len(df) < 10:
                errors.append("File must contain at least 10 rows of data")
            
            # Check for required columns (at least similar names)
            df_columns_lower = [col.lower() for col in df.columns]
            
            # Check for date column
            date_columns = [col for col in df_columns_lower if any(keyword in col for keyword in ['date', 'time', 'day'])]
            if not date_columns:
                errors.append("No date column found. Expected columns like 'date', 'datetime', 'timestamp'")
            
            # Check for product column
            product_columns = [col for col in df_columns_lower if any(keyword in col for keyword in ['product', 'item', 'sku'])]
            if not product_columns:
                errors.append("No product column found. Expected columns like 'product_id', 'product_name', 'item'")
            
            # Check for quantity/sales column
            quantity_columns = [col for col in df_columns_lower if any(keyword in col for keyword in ['quantity', 'qty', 'sales', 'sold', 'units'])]
            if not quantity_columns:
                errors.append("No quantity column found. Expected columns like 'quantity', 'sales_quantity', 'units_sold'")
            
            # Check for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                errors.append("No numeric columns found for analysis")
            
            return {
                'is_valid': len(errors) == 0,
                'errors': errors
            }
        
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def process_uploaded_data(self, df, date_col, product_col, quantity_col, price_col=None, category_col=None, store_col=None,
                            handle_duplicates=True, handle_missing='Keep as is', date_format='Auto-detect', aggregate_duplicates=True):
        """Process and standardize uploaded data"""
        try:
            processed_df = df.copy()
            
            # Rename columns to standard names
            column_mapping = {
                date_col: 'date',
                product_col: 'product_name',
                quantity_col: 'sales_quantity'
            }
            
            if price_col:
                column_mapping[price_col] = 'price'
            if category_col:
                column_mapping[category_col] = 'category'
            if store_col:
                column_mapping[store_col] = 'store_id'
            
            processed_df = processed_df.rename(columns=column_mapping)
            
            # Process date column
            if date_format == 'Auto-detect':
                processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce', infer_datetime_format=True)
            else:
                format_map = {
                    'YYYY-MM-DD': '%Y-%m-%d',
                    'MM/DD/YYYY': '%m/%d/%Y',
                    'DD/MM/YYYY': '%d/%m/%Y',
                    'YYYY/MM/DD': '%Y/%m/%d'
                }
                if date_format in format_map:
                    processed_df['date'] = pd.to_datetime(processed_df['date'], format=format_map[date_format], errors='coerce')
                else:
                    processed_df['date'] = pd.to_datetime(processed_df['date'], errors='coerce')
            
            # Remove rows with invalid dates
            processed_df = processed_df.dropna(subset=['date'])
            
            # Process product names
            processed_df['product_name'] = processed_df['product_name'].astype(str).str.strip()
            
            # Process sales quantity
            processed_df['sales_quantity'] = pd.to_numeric(processed_df['sales_quantity'], errors='coerce')
            processed_df = processed_df.dropna(subset=['sales_quantity'])
            processed_df = processed_df[processed_df['sales_quantity'] >= 0]  # Remove negative quantities
            
            # Process price if available
            if 'price' in processed_df.columns:
                processed_df['price'] = pd.to_numeric(processed_df['price'], errors='coerce')
                processed_df['price'] = processed_df['price'].fillna(0)
                processed_df['revenue'] = processed_df['sales_quantity'] * processed_df['price']
            else:
                # Estimate price and revenue if not provided
                processed_df['price'] = 10.0  # Default price
                processed_df['revenue'] = processed_df['sales_quantity'] * processed_df['price']
            
            # Process category
            if 'category' in processed_df.columns:
                processed_df['category'] = processed_df['category'].astype(str).str.strip()
                processed_df['category'] = processed_df['category'].replace(['nan', 'None'], None)
            else:
                processed_df['category'] = 'General'
            
            # Process store ID
            if 'store_id' in processed_df.columns:
                processed_df['store_id'] = processed_df['store_id'].astype(str).str.strip()
            else:
                processed_df['store_id'] = 'Store_001'
            
            # Handle missing values
            if handle_missing == 'Remove records with missing values':
                processed_df = processed_df.dropna()
            elif handle_missing == 'Fill with defaults':
                processed_df = processed_df.fillna({
                    'category': 'General',
                    'store_id': 'Store_001',
                    'price': 10.0
                })
            
            # Add additional calculated fields
            processed_df['year'] = processed_df['date'].dt.year
            processed_df['month'] = processed_df['date'].dt.month
            processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
            processed_df['is_weekend'] = (processed_df['day_of_week'] >= 5).astype(int)
            
            # Handle duplicates
            if handle_duplicates:
                processed_df = processed_df.drop_duplicates()
            
            # Aggregate duplicate date-product combinations
            if aggregate_duplicates:
                processed_df = processed_df.groupby(['date', 'product_name', 'store_id']).agg({
                    'sales_quantity': 'sum',
                    'price': 'mean',
                    'revenue': 'sum',
                    'category': 'first',
                    'year': 'first',
                    'month': 'first',
                    'day_of_week': 'first',
                    'is_weekend': 'first'
                }).reset_index()
            
            # Sort by date
            processed_df = processed_df.sort_values(['product_name', 'date'])
            
            logger.info(f"Data processed successfully. {len(processed_df)} rows after processing.")
            
            return processed_df
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def aggregate_daily_sales(self, df):
        """Aggregate sales data to daily level"""
        try:
            # Group by product and date, sum quantities and revenue
            daily_df = df.groupby(['product_name', 'date']).agg({
                'sales_quantity': 'sum',
                'revenue': 'sum',
                'category': 'first',
                'store_id': 'first'
            }).reset_index()
            
            return daily_df
        
        except Exception as e:
            logger.error(f"Error aggregating daily sales: {e}")
            return None
    
    def detect_outliers(self, df, column='sales_quantity'):
        """Detect outliers using IQR method"""
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            return {
                'outliers': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(df) * 100 if len(df) > 0 else 0,
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
        
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return None
    
    def calculate_seasonality(self, df, product_name=None):
        """Calculate seasonality patterns"""
        try:
            data = df.copy()
            
            if product_name:
                data = data[data['product_name'] == product_name]
            
            # Monthly seasonality
            monthly_avg = data.groupby(data['date'].dt.month)['sales_quantity'].mean()
            
            # Day of week seasonality
            dow_avg = data.groupby(data['date'].dt.dayofweek)['sales_quantity'].mean()
            
            # Quarterly seasonality
            quarterly_avg = data.groupby(data['date'].dt.quarter)['sales_quantity'].mean()
            
            return {
                'monthly': monthly_avg.to_dict(),
                'day_of_week': dow_avg.to_dict(),
                'quarterly': quarterly_avg.to_dict()
            }
        
        except Exception as e:
            logger.error(f"Error calculating seasonality: {e}")
            return None
    
    def clean_product_names(self, df):
        """Clean and standardize product names"""
        try:
            df_clean = df.copy()
            
            # Remove extra whitespace
            df_clean['product_name'] = df_clean['product_name'].str.strip()
            
            # Convert to title case
            df_clean['product_name'] = df_clean['product_name'].str.title()
            
            # Remove special characters (keep alphanumeric, spaces, and hyphens)
            df_clean['product_name'] = df_clean['product_name'].str.replace(r'[^\w\s\-]', '', regex=True)
            
            # Replace multiple spaces with single space
            df_clean['product_name'] = df_clean['product_name'].str.replace(r'\s+', ' ', regex=True)
            
            return df_clean
        
        except Exception as e:
            logger.error(f"Error cleaning product names: {e}")
            return df
    
    def fill_missing_dates(self, df, product_name):
        """Fill missing dates with zero sales for a product"""
        try:
            product_data = df[df['product_name'] == product_name].copy()
            
            if product_data.empty:
                return product_data
            
            # Create complete date range
            min_date = product_data['date'].min()
            max_date = product_data['date'].max()
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create complete dataframe
            complete_df = pd.DataFrame({'date': date_range})
            complete_df['product_name'] = product_name
            
            # Merge with existing data
            merged_df = complete_df.merge(product_data, on=['date', 'product_name'], how='left')
            
            # Fill missing values
            merged_df['sales_quantity'] = merged_df['sales_quantity'].fillna(0)
            merged_df['revenue'] = merged_df['revenue'].fillna(0)
            merged_df['category'] = merged_df['category'].fillna(product_data['category'].iloc[0] if not product_data.empty else 'General')
            merged_df['store_id'] = merged_df['store_id'].fillna(product_data['store_id'].iloc[0] if not product_data.empty else 'Store_001')
            
            return merged_df
        
        except Exception as e:
            logger.error(f"Error filling missing dates: {e}")
            return df[df['product_name'] == product_name]
