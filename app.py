import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from utils.database import DatabaseManager
from utils.forecasting import ForecastingEngine
from utils.data_processing import DataProcessor
from utils.inventory_optimization import InventoryOptimizer

# Page configuration
st.set_page_config(
    page_title="SmartInventory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def init_components():
    """Initialize all components"""
    db_manager = DatabaseManager()
    forecasting_engine = ForecastingEngine()
    data_processor = DataProcessor()
    inventory_optimizer = InventoryOptimizer()
    return db_manager, forecasting_engine, data_processor, inventory_optimizer

def format_currency(amount):
    """Format currency values"""
    try:
        return f"${amount:,.2f}"
    except:
        return "$0.00"

def main():
    # Initialize components
    db_manager, forecasting_engine, data_processor, inventory_optimizer = init_components()

    # Store in session state for access from other pages
    st.session_state.db_manager = db_manager
    st.session_state.forecasting_engine = forecasting_engine
    st.session_state.data_processor = data_processor
    st.session_state.inventory_optimizer = inventory_optimizer

    # Header
    st.title("🏪SmartInventory - Retail Forecasting Platform")
     st.markdown("### ~ [Varshith Sriramoju](https://www.linkedin.com/in/varshith-sriramoju-58141221b/)")
    st.markdown("**Advanced inventory management and demand forecasting for retail businesses**")

    st.markdown("---")

    # Main dashboard content
    show_dashboard()

def show_dashboard():
    """Show main dashboard"""
    db_manager = st.session_state.db_manager

    # Get summary statistics
    summary_data = db_manager.get_summary_statistics()

    if summary_data is None:
        st.info("👋 Welcome to SmartInventory! Upload your historical sales data to get started.")

        # Show getting started guide
        st.subheader("🚀 Getting Started")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 1️⃣ Upload Data
            Navigate to **Data Upload** to upload your historical sales data in CSV or Excel format.
            """)

        with col2:
            st.markdown("""
            ### 2️⃣ Explore Trends
            Use **Data Exploration** to analyze sales patterns and identify trends.
            """)

        with col3:
            st.markdown("""
            ### 3️⃣ Generate Forecasts
            Create ML-powered forecasts and get inventory recommendations.
            """)

        # Sample data option
        st.subheader("📊 Try with Sample Data")
        if st.button("Load Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                sample_data = generate_sample_data()
                if db_manager.save_sales_data(sample_data):
                    st.success("Sample data loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load sample data.")

        return

    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=format_currency(summary_data.get('total_revenue', 0)),
            delta=f"{summary_data.get('revenue_growth', 0):.1f}%"
        )

    with col2:
        st.metric(
            label="📦 Active Products",
            value=f"{summary_data.get('active_products', 0):,}",
            delta=f"{summary_data.get('new_products', 0)} new"
        )

    with col3:
        st.metric(
            label="🎯 Forecast Accuracy",
            value=f"{summary_data.get('forecast_accuracy', 0):.1f}%",
            delta=f"{summary_data.get('accuracy_trend', 0):.1f}%"
        )

    with col4:
        st.metric(
            label="⚠️ Low Stock Items",
            value=f"{summary_data.get('low_stock_items', 0):,}",
            delta=f"{summary_data.get('stock_trend', 0)} vs last week"
        )

    st.markdown("---")

    # Charts and data
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Sales Overview")

        # Get recent sales data for chart
        sales_data = db_manager.get_sales_data()
        if not sales_data.empty:
            # Daily sales trend
            daily_sales = sales_data.groupby('date').agg({
                'sales_quantity': 'sum',
                'revenue': 'sum'
            }).reset_index()

            # Limit to last 30 days for performance
            if len(daily_sales) > 30:
                daily_sales = daily_sales.tail(30)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=daily_sales['date'],
                y=daily_sales['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title='Daily Revenue Trend (Last 30 Days)',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No sales data available for visualization.")

    with col2:
        st.subheader("🚨 Alerts & Notifications")

        # Get inventory alerts
        alerts = db_manager.get_inventory_alerts()
        if not alerts.empty:
            for _, alert in alerts.iterrows():
                if alert['alert_type'] == 'low_stock':
                    st.error(f"🔴 **Low Stock**: {alert['product_name']} ({alert['current_stock']} units)")
                elif alert['alert_type'] == 'overstock':
                    st.warning(f"🟡 **Overstock**: {alert['product_name']} ({alert['current_stock']} units)")
        else:
            st.success("✅ No critical alerts")

        st.subheader("🔮 Recent Forecasts")
        recent_forecasts = db_manager.get_recent_forecasts(5)
        if not recent_forecasts.empty:
            for _, forecast in recent_forecasts.iterrows():
                st.markdown(f"**{forecast['product_name']}**: {forecast['predicted_quantity']:.1f} units")
        else:
            st.info("No recent forecasts available")

def generate_sample_data():
    """Generate sample sales data"""
    np.random.seed(42)

    products = [
        'Laptop Pro 15"', 'Wireless Headphones', 'Smartphone X',
        'Coffee Maker Deluxe', 'Running Shoes', 'Desk Lamp LED',
        'Backpack Travel', 'Water Bottle Steel', 'Bluetooth Speaker',
        'Tablet 10"', 'Gaming Mouse', 'Office Chair'
    ]

    categories = ['Electronics', 'Home', 'Sports', 'Office']

    # Generate 6 months of data
    start_date = datetime.now() - timedelta(days=180)
    data = []

    for i in range(180):
        date = start_date + timedelta(days=i)

        # Generate 5-15 transactions per day
        num_transactions = np.random.randint(5, 16)

        for _ in range(num_transactions):
            product = np.random.choice(products)
            category = np.random.choice(categories)

            # Base quantities and prices
            base_qty = np.random.randint(1, 10)
            base_price = np.random.uniform(20, 500)

            # Add seasonality
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 365)
            weekend_factor = 1.3 if date.weekday() >= 5 else 1.0

            quantity = max(1, int(base_qty * seasonal_factor * weekend_factor))
            price = round(base_price, 2)
            revenue = quantity * price

            data.append({
                'date': date.date(),
                'product_name': product,
                'category': category,
                'store_id': 'Store_001',
                'sales_quantity': quantity,
                'price': price,
                'revenue': revenue
            })

    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
