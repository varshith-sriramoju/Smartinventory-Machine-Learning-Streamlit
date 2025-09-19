import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_currency(amount, currency_symbol="$"):
    """Format currency values"""
    try:
        return f"{currency_symbol}{amount:,.2f}"
    except:
        return f"{currency_symbol}0.00"

def format_number(number, decimal_places=2):
    """Format numbers with commas and decimal places"""
    try:
        return f"{number:,.{decimal_places}f}"
    except:
        return "0.00"

def format_percentage(value, decimal_places=1):
    """Format percentage values"""
    try:
        return f"{value:.{decimal_places}f}%"
    except:
        return "0.0%"

def calculate_accuracy_metrics(actual, predicted):
    """Calculate various accuracy metrics"""
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Remove any NaN or infinite values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {
                'mape': 0,
                'rmse': 0,
                'mae': 0,
                'accuracy': 0
            }
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Accuracy (100 - MAPE)
        accuracy = max(0, 100 - mape)
        
        return {
            'mape': round(mape, 2),
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'accuracy': round(accuracy, 2)
        }
    
    except Exception as e:
        logger.error(f"Error calculating accuracy metrics: {e}")
        return {
            'mape': 0,
            'rmse': 0,
            'mae': 0,
            'accuracy': 0
        }

def calculate_trend(data, periods=7):
    """Calculate trend over specified periods"""
    try:
        if len(data) < periods + 1:
            return 0
        
        recent_avg = np.mean(data[-periods:])
        previous_avg = np.mean(data[-(periods*2):-periods])
        
        if previous_avg == 0:
            return 0
        
        trend = ((recent_avg - previous_avg) / previous_avg) * 100
        return round(trend, 2)
    
    except Exception as e:
        logger.error(f"Error calculating trend: {e}")
        return 0

def validate_date_range(start_date, end_date):
    """Validate date range inputs"""
    try:
        if start_date > end_date:
            return False, "Start date cannot be after end date"
        
        if (end_date - start_date).days > 3650:  # 10 years
            return False, "Date range cannot exceed 10 years"
        
        return True, "Valid date range"
    
    except Exception as e:
        return False, f"Invalid dates: {str(e)}"

def clean_column_names(df):
    """Clean and standardize column names"""
    try:
        df_clean = df.copy()
        
        # Convert to lowercase and replace spaces/special chars with underscores
        df_clean.columns = df_clean.columns.str.lower().str.replace(r'[^\w]', '_', regex=True)
        
        # Remove multiple underscores
        df_clean.columns = df_clean.columns.str.replace(r'_+', '_', regex=True)
        
        # Remove leading/trailing underscores
        df_clean.columns = df_clean.columns.str.strip('_')
        
        return df_clean
    
    except Exception as e:
        logger.error(f"Error cleaning column names: {e}")
        return df

def detect_date_format(date_string):
    """Detect date format from string"""
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y/%m/%d',
        '%d-%m-%Y',
        '%m-%d-%Y',
        '%Y%m%d',
        '%d.%m.%Y',
        '%m.%d.%Y'
    ]
    
    for fmt in formats:
        try:
            datetime.strptime(str(date_string), fmt)
            return fmt
        except ValueError:
            continue
    
    return None

def generate_sample_data(num_products=10, num_days=365):
    """Generate sample sales data for testing"""
    try:
        np.random.seed(42)  # For reproducible results
        
        products = [f"Product_{i+1:03d}" for i in range(num_products)]
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home']
        stores = ['Store_001', 'Store_002', 'Store_003']
        
        start_date = datetime.now() - timedelta(days=num_days)
        
        data = []
        
        for product in products:
            category = np.random.choice(categories)
            store = np.random.choice(stores)
            base_demand = np.random.randint(5, 50)
            price = np.random.uniform(10, 100)
            
            for i in range(num_days):
                date = start_date + timedelta(days=i)
                
                # Add seasonality and randomness
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
                weekly_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 7)     # Weekly seasonality
                random_factor = np.random.normal(1, 0.2)
                
                quantity = max(0, int(base_demand * seasonal_factor * weekly_factor * random_factor))
                revenue = quantity * price
                
                data.append({
                    'date': date.date(),
                    'product_name': product,
                    'category': category,
                    'store_id': store,
                    'sales_quantity': quantity,
                    'price': round(price, 2),
                    'revenue': round(revenue, 2)
                })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return pd.DataFrame()

def calculate_business_metrics(sales_data):
    """Calculate key business metrics from sales data"""
    try:
        if sales_data.empty:
            return {}
        
        # Basic metrics
        total_revenue = sales_data['revenue'].sum() if 'revenue' in sales_data.columns else 0
        total_quantity = sales_data['sales_quantity'].sum()
        avg_price = sales_data['price'].mean() if 'price' in sales_data.columns else 0
        
        # Time-based metrics
        date_range = (sales_data['date'].max() - sales_data['date'].min()).days
        avg_daily_revenue = total_revenue / max(1, date_range)
        avg_daily_quantity = total_quantity / max(1, date_range)
        
        # Product metrics
        unique_products = sales_data['product_name'].nunique()
        top_product = sales_data.groupby('product_name')['revenue'].sum().idxmax() if total_revenue > 0 else 'N/A'
        
        return {
            'total_revenue': total_revenue,
            'total_quantity': total_quantity,
            'avg_price': avg_price,
            'date_range_days': date_range,
            'avg_daily_revenue': avg_daily_revenue,
            'avg_daily_quantity': avg_daily_quantity,
            'unique_products': unique_products,
            'top_product': top_product
        }
    
    except Exception as e:
        logger.error(f"Error calculating business metrics: {e}")
        return {}

def export_to_csv(dataframe, filename_prefix="export"):
    """Export dataframe to CSV with timestamp"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        csv_string = dataframe.to_csv(index=False)
        return csv_string, filename
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return "", "export_error.csv"

def validate_forecast_inputs(forecast_days, model_type, selected_products):
    """Validate forecast generation inputs"""
    errors = []
    
    if forecast_days < 1 or forecast_days > 365:
        errors.append("Forecast days must be between 1 and 365")
    
    if not model_type:
        errors.append("Model type must be selected")
    
    if not selected_products:
        errors.append("At least one product must be selected")
    
    return len(errors) == 0, errors

def calculate_inventory_turnover(sales_data, inventory_data):
    """Calculate inventory turnover ratio"""
    try:
        if sales_data.empty or inventory_data.empty:
            return {}
        
        # Calculate cost of goods sold (using revenue as proxy)
        annual_cogs = sales_data['revenue'].sum() * 0.7  # Assume 70% of revenue is COGS
        
        # Calculate average inventory value
        avg_inventory_value = inventory_data['current_stock'].sum() * sales_data['price'].mean()
        
        # Calculate turnover ratio
        turnover_ratio = annual_cogs / max(avg_inventory_value, 1)
        
        return {
            'turnover_ratio': round(turnover_ratio, 2),
            'days_in_inventory': round(365 / max(turnover_ratio, 1), 2),
            'annual_cogs': annual_cogs,
            'avg_inventory_value': avg_inventory_value
        }
    
    except Exception as e:
        logger.error(f"Error calculating inventory turnover: {e}")
        return {}
