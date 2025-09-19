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

st.set_page_config(
    page_title="Performance Metrics - SmartInventory",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize database manager"""
    return DatabaseManager()

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
            return {'mape': 0, 'rmse': 0, 'mae': 0, 'accuracy': 0}
        
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
        return {'mape': 0, 'rmse': 0, 'mae': 0, 'accuracy': 0}

def main():
    st.title("📊 Performance Metrics")
    st.markdown("Monitor and analyze forecasting model performance")
    
    db = init_components()
    
    # Get model performance data
    performance_data = db.get_model_performance()
    
    if performance_data is None or performance_data.empty:
        st.warning("📈 No model performance data available. Generate some forecasts first.")
        
        st.subheader("Getting Started")
        st.markdown("""
        1. Navigate to the **Forecasting** page
        2. Generate forecasts for your products
        3. Return here to view model performance metrics
        """)
        return
    
    st.success(f"✅ Loaded performance data for {len(performance_data)} forecast runs")
    
    # Performance metrics overview
    st.header("🎯 Overall Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_accuracy = 100 - performance_data['mape'].mean()
        st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
    
    with col2:
        avg_mape = performance_data['mape'].mean()
        st.metric("Average MAPE", f"{avg_mape:.2f}%")
    
    with col3:
        avg_rmse = performance_data['rmse'].mean()
        st.metric("Average RMSE", f"{avg_rmse:.2f}")
    
    with col4:
        total_forecasts = len(performance_data)
        st.metric("Total Forecasts", f"{total_forecasts:,}")
    
    st.divider()
    
    # Performance trends and analysis
    tabs = st.tabs(["📈 Trends", "🏆 Product Rankings", "🔬 Model Comparison", "📋 Detailed Data"])
    
    with tabs[0]:
        # Performance trends over time
        st.subheader("📈 Performance Trends Over Time")
        
        if 'created_at' in performance_data.columns:
            # Daily performance trends
            daily_performance = performance_data.groupby(performance_data['created_at'].dt.date).agg({
                'mape': 'mean',
                'rmse': 'mean',
                'mae': 'mean'
            }).reset_index()
            daily_performance['accuracy'] = 100 - daily_performance['mape']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Accuracy Over Time', 'MAPE Over Time', 'RMSE Over Time', 'MAE Over Time'],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Accuracy trend
            fig.add_trace(
                go.Scatter(
                    x=daily_performance['created_at'],
                    y=daily_performance['accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # MAPE trend
            fig.add_trace(
                go.Scatter(
                    x=daily_performance['created_at'],
                    y=daily_performance['mape'],
                    mode='lines+markers',
                    name='MAPE',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
            
            # RMSE trend
            fig.add_trace(
                go.Scatter(
                    x=daily_performance['created_at'],
                    y=daily_performance['rmse'],
                    mode='lines+markers',
                    name='RMSE',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # MAE trend
            fig.add_trace(
                go.Scatter(
                    x=daily_performance['created_at'],
                    y=daily_performance['mae'],
                    mode='lines+markers',
                    name='MAE',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Model Performance Trends"
            )
            
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
            fig.update_yaxes(title_text="MAPE (%)", row=1, col=2)
            fig.update_yaxes(title_text="RMSE", row=2, col=1)
            fig.update_yaxes(title_text="MAE", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timestamp data available for trend analysis")
        
        # Performance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                performance_data,
                x='mape',
                nbins=20,
                title='Distribution of MAPE Values',
                labels={'mape': 'MAPE (%)', 'count': 'Number of Forecasts'}
            )
            fig.add_vline(x=performance_data['mape'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {performance_data['mape'].mean():.2f}%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            performance_data['accuracy'] = 100 - performance_data['mape']
            fig = px.histogram(
                performance_data,
                x='accuracy',
                nbins=20,
                title='Distribution of Accuracy Values',
                labels={'accuracy': 'Accuracy (%)', 'count': 'Number of Forecasts'}
            )
            fig.add_vline(x=performance_data['accuracy'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {performance_data['accuracy'].mean():.1f}%")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Product-level performance rankings
        st.subheader("🏆 Product Performance Rankings")
        
        product_performance = performance_data.groupby('product_name').agg({
            'mape': 'mean',
            'rmse': 'mean',
            'mae': 'mean',
            'product_name': 'count'  # Count forecasts per product
        }).round(2)
        product_performance.columns = ['Avg_MAPE', 'Avg_RMSE', 'Avg_MAE', 'Forecast_Count']
        product_performance['Avg_Accuracy'] = (100 - product_performance['Avg_MAPE']).round(1)
        product_performance = product_performance.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Best Performing Products (Highest Accuracy)")
            best_products = product_performance.nlargest(10, 'Avg_Accuracy')
            
            st.dataframe(
                best_products[['product_name', 'Avg_Accuracy', 'Avg_MAPE', 'Forecast_Count']],
                column_config={
                    'product_name': 'Product',
                    'Avg_Accuracy': st.column_config.NumberColumn('Accuracy (%)', format="%.1f"),
                    'Avg_MAPE': st.column_config.NumberColumn('MAPE (%)', format="%.2f"),
                    'Forecast_Count': 'Forecasts'
                },
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.subheader("⚠️ Products Needing Attention (Lowest Accuracy)")
            worst_products = product_performance.nsmallest(10, 'Avg_Accuracy')
            
            st.dataframe(
                worst_products[['product_name', 'Avg_Accuracy', 'Avg_MAPE', 'Forecast_Count']],
                column_config={
                    'product_name': 'Product',
                    'Avg_Accuracy': st.column_config.NumberColumn('Accuracy (%)', format="%.1f"),
                    'Avg_MAPE': st.column_config.NumberColumn('MAPE (%)', format="%.2f"),
                    'Forecast_Count': 'Forecasts'
                },
                use_container_width=True,
                hide_index=True
            )
        
        # Product performance visualization
        fig = px.scatter(
            product_performance,
            x='Avg_MAPE',
            y='Forecast_Count',
            size='Avg_Accuracy',
            hover_data=['product_name'],
            title='Product Performance: MAPE vs Forecast Count (Bubble size = Accuracy)',
            labels={'Avg_MAPE': 'Average MAPE (%)', 'Forecast_Count': 'Number of Forecasts'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Model comparison
        st.subheader("🔬 Model Performance Comparison")
        
        if 'model_type' in performance_data.columns:
            model_performance = performance_data.groupby('model_type').agg({
                'mape': ['mean', 'std', 'count'],
                'rmse': ['mean', 'std'],
                'mae': ['mean', 'std']
            }).round(3)
            
            # Flatten column names
            model_performance.columns = ['_'.join(col).strip() for col in model_performance.columns]
            model_performance = model_performance.reset_index()
            
            # Calculate accuracy
            model_performance['accuracy_mean'] = 100 - model_performance['mape_mean']
            
            # Display model comparison table
            st.dataframe(
                model_performance[['model_type', 'accuracy_mean', 'mape_mean', 'mape_std', 'rmse_mean', 'mae_mean', 'mape_count']],
                column_config={
                    'model_type': 'Model',
                    'accuracy_mean': st.column_config.NumberColumn('Avg Accuracy (%)', format="%.1f"),
                    'mape_mean': st.column_config.NumberColumn('Avg MAPE (%)', format="%.2f"),
                    'mape_std': st.column_config.NumberColumn('MAPE Std Dev', format="%.2f"),
                    'rmse_mean': st.column_config.NumberColumn('Avg RMSE', format="%.2f"),
                    'mae_mean': st.column_config.NumberColumn('Avg MAE', format="%.2f"),
                    'mape_count': 'Forecasts'
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Model comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    model_performance,
                    x='model_type',
                    y='accuracy_mean',
                    title='Average Accuracy by Model Type',
                    labels={'accuracy_mean': 'Average Accuracy (%)', 'model_type': 'Model Type'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    performance_data,
                    x='model_type',
                    y='mape',
                    title='MAPE Distribution by Model Type',
                    labels={'mape': 'MAPE (%)', 'model_type': 'Model Type'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Model usage over time
            if 'created_at' in performance_data.columns:
                model_usage = performance_data.groupby([
                    performance_data['created_at'].dt.date, 'model_type'
                ]).size().reset_index(name='count')
                model_usage.columns = ['date', 'model_type', 'count']
                
                fig = px.line(
                    model_usage,
                    x='date',
                    y='count',
                    color='model_type',
                    title='Model Usage Over Time',
                    labels={'count': 'Number of Forecasts', 'date': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model type data available for comparison")
    
    with tabs[3]:
        # Detailed performance data
        st.subheader("📋 Detailed Performance Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            models = ['All'] + list(performance_data['model_type'].unique()) if 'model_type' in performance_data.columns else ['All']
            selected_model = st.selectbox("Filter by Model", models)
        
        with col2:
            products = ['All'] + list(performance_data['product_name'].unique())
            selected_product = st.selectbox("Filter by Product", products)
        
        with col3:
            min_accuracy = st.slider("Minimum Accuracy (%)", 0, 100, 0)
        
        # Apply filters
        filtered_data = performance_data.copy()
        
        if selected_model != 'All' and 'model_type' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['model_type'] == selected_model]
        
        if selected_product != 'All':
            filtered_data = filtered_data[filtered_data['product_name'] == selected_product]
        
        filtered_data['accuracy'] = 100 - filtered_data['mape']
        filtered_data = filtered_data[filtered_data['accuracy'] >= min_accuracy]
        
        # Display filtered data
        display_columns = ['product_name', 'mape', 'rmse', 'mae', 'accuracy']
        if 'model_type' in filtered_data.columns:
            display_columns.insert(1, 'model_type')
        if 'created_at' in filtered_data.columns:
            display_columns.append('created_at')
        
        st.dataframe(
            filtered_data[display_columns],
            column_config={
                'product_name': 'Product',
                'model_type': 'Model',
                'mape': st.column_config.NumberColumn('MAPE (%)', format="%.2f"),
                'rmse': st.column_config.NumberColumn('RMSE', format="%.2f"),
                'mae': st.column_config.NumberColumn('MAE', format="%.2f"),
                'accuracy': st.column_config.NumberColumn('Accuracy (%)', format="%.1f"),
                'created_at': st.column_config.DatetimeColumn('Created At')
            },
            use_container_width=True,
            height=400
        )
        
        # Export option
        st.divider()
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("📥 Export Performance Data", type="secondary"):
                csv = filtered_data.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"model_performance_{timestamp}.csv",
                    mime="text/csv"
                )
    
    # Performance insights
    st.divider()
    st.header("💡 Performance Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Key Statistics")
        
        insights = []
        
        # Best and worst accuracy
        best_accuracy = performance_data['mape'].min()
        worst_accuracy = performance_data['mape'].max()
        insights.append(f"• Best forecast accuracy: {100 - best_accuracy:.1f}%")
        insights.append(f"• Worst forecast accuracy: {100 - worst_accuracy:.1f}%")
        
        # Accuracy distribution
        high_accuracy = len(performance_data[performance_data['mape'] < 10])
        total_forecasts = len(performance_data)
        insights.append(f"• High accuracy forecasts (>90%): {high_accuracy}/{total_forecasts} ({high_accuracy/total_forecasts*100:.1f}%)")
        
        # Most/least reliable products
        if len(product_performance) > 0:
            most_reliable = product_performance.loc[product_performance['Avg_Accuracy'].idxmax(), 'product_name']
            least_reliable = product_performance.loc[product_performance['Avg_Accuracy'].idxmin(), 'product_name']
            insights.append(f"• Most reliable product: {most_reliable}")
            insights.append(f"• Least reliable product: {least_reliable}")
        
        for insight in insights:
            st.markdown(insight)
    
    with col2:
        st.subheader("📈 Recommendations")
        
        recommendations = []
        
        avg_mape = performance_data['mape'].mean()
        if avg_mape > 20:
            recommendations.append("🔴 Overall accuracy is low. Consider data quality improvements.")
        elif avg_mape > 10:
            recommendations.append("🟡 Accuracy is moderate. Look for patterns in poorly performing products.")
        else:
            recommendations.append("🟢 Good overall accuracy. Continue monitoring for consistency.")
        
        # Check for model performance differences
        if 'model_type' in performance_data.columns and len(performance_data['model_type'].unique()) > 1:
            model_comparison = performance_data.groupby('model_type')['mape'].mean()
            best_model = model_comparison.idxmin()
            recommendations.append(f"💡 {best_model} performs best on average. Consider using it more frequently.")
        
        # Check for products needing attention
        if len(product_performance) > 0:
            poor_products = product_performance[product_performance['Avg_Accuracy'] < 70]
            if len(poor_products) > 0:
                recommendations.append(f"⚠️ {len(poor_products)} products have accuracy below 70%. Consider additional features or data.")
        
        recommendations.append("📊 Regular monitoring of these metrics helps maintain forecast quality.")
        
        for recommendation in recommendations:
            st.markdown(recommendation)

if __name__ == "__main__":
    main()
