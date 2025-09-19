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
from utils.forecasting import ForecastingEngine

st.set_page_config(
    page_title="Forecasting - SmartInventory",
    page_icon="🔮",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize components"""
    db = DatabaseManager()
    forecasting_engine = ForecastingEngine()
    return db, forecasting_engine

def main():
    st.title("🔮 Sales Forecasting")
    st.markdown("Generate ML-powered demand forecasts for your products")
    
    db, forecasting_engine = init_components()
    
    # Check if data exists
    sales_data = db.get_sales_data()
    
    if sales_data is None or sales_data.empty:
        st.warning("📥 No sales data available. Please upload data first using the **Data Upload** page.")
        return
    
    # Get unique products
    products = sorted(sales_data['product_name'].unique().tolist())
    
    st.success(f"✅ Ready to forecast for {len(products)} products")
    
    # Forecasting configuration
    st.header("⚙️ Forecast Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_products = st.multiselect(
            "Select Products for Forecasting",
            products,
            default=products[:min(5, len(products))],
            help="Select up to 10 products for forecasting"
        )
    
    with col2:
        forecast_days = st.number_input(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=30,
            help="Number of days to forecast into the future"
        )
    
    with col3:
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "Linear Regression", "ARIMA"],
            help="Choose the forecasting algorithm"
        )
    
    with col4:
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            help="Confidence level for prediction intervals"
        )
    
    # Validation
    if len(selected_products) > 10:
        st.error("⚠️ Please select maximum 10 products for performance reasons.")
        return
    
    if not selected_products:
        st.warning("⚠️ Please select at least one product to forecast.")
        return
    
    # Generate forecasts
    if st.button("🚀 Generate Forecasts", type="primary", use_container_width=True):
        
        with st.spinner("Generating forecasts... This may take a few minutes."):
            forecasts = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, product in enumerate(selected_products):
                status_text.text(f"Processing {product}... ({i+1}/{len(selected_products)})")
                progress_bar.progress((i + 1) / len(selected_products))
                
                # Get product data
                product_data = sales_data[sales_data['product_name'] == product].copy()
                
                # Check minimum data requirement
                if len(product_data) < 30:
                    st.warning(f"⚠️ Insufficient data for {product} (minimum 30 days required, found {len(product_data)} days)")
                    continue
                
                # Generate forecast
                try:
                    forecast_result = forecasting_engine.generate_forecast(
                        product_data, forecast_days, model_type, confidence_level
                    )
                    
                    if forecast_result is not None:
                        forecasts[product] = forecast_result
                        
                        # Save forecast to database
                        db.save_forecast(product, forecast_result)
                        
                        # Save model performance
                        db.save_model_performance(
                            product, model_type,
                            forecast_result.get('mape', 0),
                            forecast_result.get('rmse', 0),
                            forecast_result.get('mae', 0)
                        )
                    else:
                        st.warning(f"⚠️ Could not generate forecast for {product}")
                
                except Exception as e:
                    st.error(f"❌ Error forecasting {product}: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
            
            if forecasts:
                st.success(f"✅ Successfully generated forecasts for {len(forecasts)} products!")
                
                # Display forecasts
                st.header("📈 Forecast Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                avg_accuracy = np.mean([f.get('accuracy', 0) for f in forecasts.values()])
                total_forecast = sum([f['forecast']['predicted_quantity'].sum() for f in forecasts.values()])
                avg_mape = np.mean([f.get('mape', 0) for f in forecasts.values()])
                
                with col1:
                    st.metric("Forecasts Generated", len(forecasts))
                with col2:
                    st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
                with col3:
                    st.metric("Total Forecasted Units", f"{total_forecast:,.0f}")
                with col4:
                    st.metric("Average MAPE", f"{avg_mape:.1f}%")
                
                st.divider()
                
                # Individual product forecasts
                for product, forecast_data in forecasts.items():
                    with st.expander(f"📦 {product} - Forecast Details", expanded=True):
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Create forecast plot
                            fig = go.Figure()
                            
                            # Historical data
                            historical = forecast_data['historical']
                            fig.add_trace(go.Scatter(
                                x=historical['date'],
                                y=historical['sales_quantity'],
                                mode='lines+markers',
                                name='Historical Sales',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=4)
                            ))
                            
                            # Forecast data
                            forecast = forecast_data['forecast']
                            fig.add_trace(go.Scatter(
                                x=forecast['date'],
                                y=forecast['predicted_quantity'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=4)
                            ))
                            
                            # Confidence intervals
                            if 'confidence_lower' in forecast.columns and 'confidence_upper' in forecast.columns:
                                fig.add_trace(go.Scatter(
                                    x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                                    y=forecast['confidence_upper'].tolist() + forecast['confidence_lower'].tolist()[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,127,14,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name=f'{confidence_level}% Confidence Interval',
                                    showlegend=True
                                ))
                            
                            fig.update_layout(
                                title=f'Sales Forecast for {product}',
                                xaxis_title='Date',
                                yaxis_title='Sales Quantity',
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Forecast metrics
                            st.subheader("📊 Forecast Metrics")
                            
                            avg_forecast = forecast['predicted_quantity'].mean()
                            total_forecast = forecast['predicted_quantity'].sum()
                            max_forecast = forecast['predicted_quantity'].max()
                            min_forecast = forecast['predicted_quantity'].min()
                            
                            st.metric("Average Daily Forecast", f"{avg_forecast:.1f}")
                            st.metric("Total Period Forecast", f"{total_forecast:.0f}")
                            st.metric("Peak Day Forecast", f"{max_forecast:.1f}")
                            st.metric("Minimum Day Forecast", f"{min_forecast:.1f}")
                            
                            st.subheader("🎯 Model Performance")
                            st.metric("Accuracy", f"{forecast_data.get('accuracy', 0):.1f}%")
                            st.metric("MAPE", f"{forecast_data.get('mape', 0):.2f}%")
                            st.metric("RMSE", f"{forecast_data.get('rmse', 0):.2f}")
                            st.metric("MAE", f"{forecast_data.get('mae', 0):.2f}")
                        
                        # Forecast data table
                        st.subheader("📋 Detailed Forecast Data")
                        
                        display_forecast = forecast[['date', 'predicted_quantity']].copy()
                        display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m-%d')
                        display_forecast['predicted_quantity'] = display_forecast['predicted_quantity'].round(2)
                        display_forecast.columns = ['Date', 'Predicted Quantity']
                        
                        st.dataframe(display_forecast, use_container_width=True, height=200)
                        
                        # Export individual forecast
                        csv = display_forecast.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label=f"📥 Export {product} Forecast",
                            data=csv,
                            file_name=f"forecast_{product}_{timestamp}.csv",
                            mime="text/csv",
                            key=f"export_{product}"
                        )
                
                # Export all forecasts
                st.divider()
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export All Forecasts", type="secondary"):
                        all_forecasts = []
                        for product, forecast_data in forecasts.items():
                            forecast_df = forecast_data['forecast'][['date', 'predicted_quantity']].copy()
                            forecast_df['product_name'] = product
                            forecast_df['model_type'] = forecast_data.get('model_type', model_type)
                            forecast_df['accuracy'] = forecast_data.get('accuracy', 0)
                            all_forecasts.append(forecast_df)
                        
                        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
                        csv = combined_forecasts.to_csv(index=False)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="Download All Forecasts CSV",
                            data=csv,
                            file_name=f"all_forecasts_{timestamp}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if st.button("📊 View Performance Summary", type="secondary"):
                        st.subheader("🎯 Model Performance Summary")
                        
                        performance_data = []
                        for product, forecast_data in forecasts.items():
                            performance_data.append({
                                'Product': product,
                                'Model': forecast_data.get('model_type', model_type),
                                'Accuracy (%)': f"{forecast_data.get('accuracy', 0):.1f}",
                                'MAPE (%)': f"{forecast_data.get('mape', 0):.2f}",
                                'RMSE': f"{forecast_data.get('rmse', 0):.2f}",
                                'MAE': f"{forecast_data.get('mae', 0):.2f}"
                            })
                        
                        performance_df = pd.DataFrame(performance_data)
                        st.dataframe(performance_df, use_container_width=True)
                
                with col3:
                    if st.button("🔄 Generate New Forecasts", type="secondary"):
                        st.rerun()
            else:
                st.error("❌ No forecasts could be generated. Please check your data and try again.")
    
    # Recent forecasts section
    st.divider()
    st.header("📊 Recent Forecasts")
    
    recent_forecasts = db.get_recent_forecasts(20)
    if not recent_forecasts.empty:
        # Group by product and show latest forecast for each
        latest_forecasts = recent_forecasts.groupby('product_name').first().reset_index()
        
        st.dataframe(
            latest_forecasts[['product_name', 'forecast_date', 'predicted_quantity', 'model_type', 'created_at']],
            column_config={
                'product_name': 'Product',
                'forecast_date': 'Forecast Date',
                'predicted_quantity': st.column_config.NumberColumn('Predicted Quantity', format="%.2f"),
                'model_type': 'Model',
                'created_at': st.column_config.DatetimeColumn('Generated At')
            },
            use_container_width=True
        )
    else:
        st.info("No recent forecasts available. Generate some forecasts to see them here.")

if __name__ == "__main__":
    main()
