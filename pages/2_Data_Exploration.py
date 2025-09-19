import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from utils.visualizations import create_interactive_charts

st.set_page_config(
    page_title="Data Exploration - SmartInventory",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize database manager"""
    return DatabaseManager()

@st.cache_data
def load_data():
    """Load data from database with caching"""
    db = DatabaseManager()
    return db.get_sales_data()

def main():
    st.title("📊 Data Exploration")
    st.markdown("Analyze sales patterns, trends, and insights from your data")
    
    db = init_components()
    
    # Load data
    sales_data = load_data()
    
    if sales_data is None or sales_data.empty:
        st.warning("📥 No sales data available. Please upload data first using the **Data Upload** page.")
        
        # Show data upload instructions
        st.subheader("Getting Started")
        st.markdown("""
        1. Navigate to the **Data Upload** page
        2. Upload your historical sales data (CSV or Excel)
        3. Return here to explore your data
        """)
        return
    
    st.success(f"✅ Loaded {len(sales_data):,} records for analysis")
    
    # Data overview
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = sales_data['revenue'].sum() if 'revenue' in sales_data.columns else 0
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_units = sales_data['sales_quantity'].sum()
        st.metric("Total Units Sold", f"{total_units:,.0f}")
    
    with col3:
        unique_products = sales_data['product_name'].nunique()
        st.metric("Unique Products", f"{unique_products:,}")
    
    with col4:
        date_range = (sales_data['date'].max() - sales_data['date'].min()).days
        st.metric("Date Range", f"{date_range} days")
    
    st.divider()
    
    # Filters
    st.header("🔍 Filters")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        products = ['All'] + sorted(sales_data['product_name'].unique().tolist())
        selected_product = st.selectbox("Product", products)
    
    with col2:
        categories = ['All']
        if 'category' in sales_data.columns:
            categories += sorted(sales_data['category'].dropna().unique().tolist())
        selected_category = st.selectbox("Category", categories)
    
    with col3:
        min_date = sales_data['date'].min()
        max_date = sales_data['date'].max()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    with col4:
        stores = ['All']
        if 'store_id' in sales_data.columns:
            stores += sorted(sales_data['store_id'].dropna().unique().tolist())
        selected_store = st.selectbox("Store", stores)
    
    # Apply filters
    filtered_data = sales_data.copy()
    
    if selected_product != 'All':
        filtered_data = filtered_data[filtered_data['product_name'] == selected_product]
    
    if selected_category != 'All' and 'category' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['category'] == selected_category]
    
    if selected_store != 'All' and 'store_id' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['store_id'] == selected_store]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['date'] >= pd.Timestamp(start_date)) &
            (filtered_data['date'] <= pd.Timestamp(end_date))
        ]
    
    if filtered_data.empty:
        st.warning("No data matches the selected filters. Please adjust your selection.")
        return
    
    st.info(f"Showing {len(filtered_data):,} records after filtering")
    
    st.divider()
    
    # Time series analysis
    st.header("📈 Time Series Analysis")
    
    # Daily aggregation
    daily_data = filtered_data.groupby('date').agg({
        'sales_quantity': 'sum',
        'revenue': 'sum' if 'revenue' in filtered_data.columns else lambda x: 0
    }).reset_index()
    
    # Create time series charts
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Daily Sales Quantity', 'Daily Revenue'],
        vertical_spacing=0.1
    )
    
    # Sales quantity chart
    fig.add_trace(
        go.Scatter(
            x=daily_data['date'],
            y=daily_data['sales_quantity'],
            mode='lines+markers',
            name='Sales Quantity',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Revenue chart
    if 'revenue' in filtered_data.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Sales Trends Over Time"
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Quantity", row=1, col=1)
    fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product performance analysis
    st.header("🏆 Product Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top products by quantity
        top_products_qty = filtered_data.groupby('product_name')['sales_quantity'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_products_qty.values,
            y=top_products_qty.index,
            orientation='h',
            title='Top 10 Products by Quantity Sold',
            labels={'x': 'Quantity Sold', 'y': 'Product'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top products by revenue
        if 'revenue' in filtered_data.columns:
            top_products_rev = filtered_data.groupby('product_name')['revenue'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=top_products_rev.values,
                y=top_products_rev.index,
                orientation='h',
                title='Top 10 Products by Revenue',
                labels={'x': 'Revenue ($)', 'y': 'Product'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue data not available")
    
    # Seasonality analysis
    st.header("📅 Seasonality Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Monthly seasonality
        monthly_sales = filtered_data.groupby(filtered_data['date'].dt.month)['sales_quantity'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = px.bar(
            x=[month_names[i-1] for i in monthly_sales.index],
            y=monthly_sales.values,
            title='Sales by Month',
            labels={'x': 'Month', 'y': 'Quantity Sold'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Day of week seasonality
        dow_sales = filtered_data.groupby(filtered_data['date'].dt.dayofweek)['sales_quantity'].sum()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig = px.bar(
            x=[dow_names[i] for i in dow_sales.index],
            y=dow_sales.values,
            title='Sales by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Quantity Sold'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Quarterly seasonality
        quarterly_sales = filtered_data.groupby(filtered_data['date'].dt.quarter)['sales_quantity'].sum()
        
        fig = px.bar(
            x=[f'Q{i}' for i in quarterly_sales.index],
            y=quarterly_sales.values,
            title='Sales by Quarter',
            labels={'x': 'Quarter', 'y': 'Quantity Sold'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Category analysis (if available)
    if 'category' in filtered_data.columns and filtered_data['category'].notna().any():
        st.header("🏷️ Category Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_dist = filtered_data.groupby('category')['sales_quantity'].sum().sort_values(ascending=False)
            
            fig = px.pie(
                values=category_dist.values,
                names=category_dist.index,
                title='Sales Distribution by Category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category trends over time
            category_trends = filtered_data.groupby(['date', 'category'])['sales_quantity'].sum().reset_index()
            
            fig = px.line(
                category_trends,
                x='date',
                y='sales_quantity',
                color='category',
                title='Category Sales Trends Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical summary
    st.header("📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sales Quantity Statistics")
        qty_stats = filtered_data['sales_quantity'].describe()
        st.dataframe(qty_stats.to_frame().T, use_container_width=True)
    
    with col2:
        if 'revenue' in filtered_data.columns:
            st.subheader("Revenue Statistics")
            rev_stats = filtered_data['revenue'].describe()
            st.dataframe(rev_stats.to_frame().T, use_container_width=True)
        else:
            st.info("Revenue statistics not available")
    
    # Export data option
    st.divider()
    st.header("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Filtered Data", type="secondary"):
            csv = filtered_data.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sales_data_filtered_{timestamp}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Daily Summary", type="secondary"):
            csv = daily_data.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"daily_summary_{timestamp}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("🏆 Export Product Summary", type="secondary"):
            product_summary = filtered_data.groupby('product_name').agg({
                'sales_quantity': ['sum', 'mean', 'count'],
                'revenue': ['sum', 'mean'] if 'revenue' in filtered_data.columns else 'count'
            }).round(2)
            csv = product_summary.to_csv()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"product_summary_{timestamp}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
