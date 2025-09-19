import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_interactive_charts(data, chart_type='line', x_col='date', y_col='sales_quantity', 
                            color_col=None, title='Interactive Chart', height=400):
    """Create interactive charts using Plotly"""
    try:
        if data.empty:
            return create_empty_chart(title, height)
        
        if chart_type == 'line':
            if color_col and color_col in data.columns:
                fig = px.line(data, x=x_col, y=y_col, color=color_col, 
                            title=title, height=height)
            else:
                fig = px.line(data, x=x_col, y=y_col, title=title, height=height)
        
        elif chart_type == 'bar':
            if color_col and color_col in data.columns:
                fig = px.bar(data, x=x_col, y=y_col, color=color_col, 
                           title=title, height=height)
            else:
                fig = px.bar(data, x=x_col, y=y_col, title=title, height=height)
        
        elif chart_type == 'scatter':
            if color_col and color_col in data.columns:
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, 
                               title=title, height=height)
            else:
                fig = px.scatter(data, x=x_col, y=y_col, title=title, height=height)
        
        elif chart_type == 'histogram':
            fig = px.histogram(data, x=x_col, title=title, height=height)
        
        elif chart_type == 'box':
            fig = px.box(data, x=x_col, y=y_col, title=title, height=height)
        
        else:
            # Default to line chart
            fig = px.line(data, x=x_col, y=y_col, title=title, height=height)
        
        # Update layout for better appearance
        fig.update_layout(
            showlegend=True,
            hovermode='x unified' if chart_type == 'line' else 'closest',
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating chart: {e}")
        return create_empty_chart(title, height)

def create_empty_chart(title='No Data Available', height=400):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available for visualization",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    return fig

def create_sales_trend_chart(data, title='Sales Trend Over Time'):
    """Create a sales trend chart with multiple metrics"""
    try:
        if data.empty:
            return create_empty_chart(title)
        
        # Aggregate daily sales
        daily_data = data.groupby('date').agg({
            'sales_quantity': 'sum',
            'revenue': 'sum' if 'revenue' in data.columns else lambda x: 0
        }).reset_index()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Daily Sales Quantity', 'Daily Revenue'],
            vertical_spacing=0.1
        )
        
        # Sales quantity
        fig.add_trace(
            go.Scatter(
                x=daily_data['date'],
                y=daily_data['sales_quantity'],
                mode='lines+markers',
                name='Sales Quantity',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )
        
        # Revenue (if available)
        if 'revenue' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_data['date'],
                    y=daily_data['revenue'],
                    mode='lines+markers',
                    name='Revenue',
                    line=dict(color='#ff7f0e', width=2)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Quantity", row=1, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=2, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating sales trend chart: {e}")
        return create_empty_chart(title)

def create_product_performance_chart(data, metric='sales_quantity', top_n=10):
    """Create a horizontal bar chart for top performing products"""
    try:
        if data.empty:
            return create_empty_chart(f'Top {top_n} Products by {metric.replace("_", " ").title()}')
        
        # Aggregate by product
        product_data = data.groupby('product_name')[metric].sum().sort_values(ascending=False).head(top_n)
        
        fig = px.bar(
            x=product_data.values,
            y=product_data.index,
            orientation='h',
            title=f'Top {top_n} Products by {metric.replace("_", " ").title()}',
            labels={'x': metric.replace("_", " ").title(), 'y': 'Product'}
        )
        
        fig.update_layout(height=max(400, top_n * 40))
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating product performance chart: {e}")
        return create_empty_chart(f'Top {top_n} Products')

def create_seasonality_charts(data):
    """Create charts showing seasonality patterns"""
    try:
        if data.empty:
            return create_empty_chart('Seasonality Analysis')
        
        # Create subplots for different seasonality patterns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Monthly Pattern', 'Day of Week Pattern', 
                          'Quarterly Pattern', 'Hourly Pattern (if available)'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Monthly seasonality
        monthly_data = data.groupby(data['date'].dt.month)['sales_quantity'].sum()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Bar(x=[month_names[i-1] for i in monthly_data.index], 
                   y=monthly_data.values, 
                   name='Monthly',
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Day of week seasonality
        dow_data = data.groupby(data['date'].dt.dayofweek)['sales_quantity'].sum()
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig.add_trace(
            go.Bar(x=[dow_names[i] for i in dow_data.index], 
                   y=dow_data.values, 
                   name='Day of Week',
                   marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Quarterly seasonality
        quarterly_data = data.groupby(data['date'].dt.quarter)['sales_quantity'].sum()
        
        fig.add_trace(
            go.Bar(x=[f'Q{i}' for i in quarterly_data.index], 
                   y=quarterly_data.values, 
                   name='Quarterly',
                   marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Placeholder for hourly (using day of month instead)
        dom_data = data.groupby(data['date'].dt.day)['sales_quantity'].mean()
        
        fig.add_trace(
            go.Bar(x=dom_data.index, 
                   y=dom_data.values, 
                   name='Day of Month',
                   marker_color='lightyellow'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Seasonality Analysis',
            height=600,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating seasonality charts: {e}")
        return create_empty_chart('Seasonality Analysis')

def create_forecast_chart(historical_data, forecast_data, product_name, confidence_intervals=True):
    """Create a forecast visualization chart"""
    try:
        fig = go.Figure()
        
        # Historical data
        if not historical_data.empty:
            fig.add_trace(go.Scatter(
                x=historical_data['date'],
                y=historical_data['sales_quantity'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=4)
            ))
        
        # Forecast data
        if not forecast_data.empty:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'],
                y=forecast_data['predicted_quantity'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=4)
            ))
            
            # Confidence intervals
            if confidence_intervals and 'confidence_lower' in forecast_data.columns and 'confidence_upper' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
                    y=forecast_data['confidence_upper'].tolist() + forecast_data['confidence_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,127,14,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
        
        fig.update_layout(
            title=f'Sales Forecast for {product_name}',
            xaxis_title='Date',
            yaxis_title='Sales Quantity',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating forecast chart: {e}")
        return create_empty_chart(f'Forecast for {product_name}')

def create_inventory_dashboard_chart(recommendations_df):
    """Create inventory dashboard visualization"""
    try:
        if recommendations_df.empty:
            return create_empty_chart('Inventory Dashboard')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Current vs Safety Stock', 'Action Distribution', 
                          'Urgency Score Distribution', 'Reorder Recommendations'],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Current vs Safety Stock (top products)
        top_products = recommendations_df.head(10)
        
        fig.add_trace(
            go.Bar(name='Current Stock', 
                   x=top_products['product_name'], 
                   y=top_products['current_stock'],
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(name='Safety Stock', 
                   x=top_products['product_name'], 
                   y=top_products['safety_stock'],
                   marker_color='orange'),
            row=1, col=1
        )
        
        # Action distribution
        action_counts = recommendations_df['action'].value_counts()
        fig.add_trace(
            go.Pie(labels=action_counts.index, 
                   values=action_counts.values,
                   name="Actions"),
            row=1, col=2
        )
        
        # Urgency score distribution
        fig.add_trace(
            go.Histogram(x=recommendations_df['urgency_score'],
                        name="Urgency Scores",
                        marker_color='red',
                        opacity=0.7),
            row=2, col=1
        )
        
        # Reorder recommendations
        reorder_items = recommendations_df[recommendations_df['action'] == 'reorder'].head(10)
        if not reorder_items.empty:
            fig.add_trace(
                go.Bar(x=reorder_items['product_name'], 
                       y=reorder_items['recommended_order_quantity'],
                       name='Order Quantity',
                       marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Inventory Management Dashboard',
            height=800,
            showlegend=False
        )
        
        # Update x-axis labels for readability
        fig.update_xaxes(tickangle=45, row=1, col=1)
        fig.update_xaxes(tickangle=45, row=2, col=2)
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating inventory dashboard: {e}")
        return create_empty_chart('Inventory Dashboard')

def create_performance_metrics_chart(performance_data):
    """Create model performance metrics visualization"""
    try:
        if performance_data.empty:
            return create_empty_chart('Model Performance Metrics')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy Distribution', 'MAPE vs RMSE', 
                          'Performance by Model Type', 'Performance Trend'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "box"}, {"type": "scatter"}]]
        )
        
        # Accuracy distribution
        performance_data['accuracy'] = 100 - performance_data['mape']
        fig.add_trace(
            go.Histogram(x=performance_data['accuracy'],
                        name="Accuracy",
                        marker_color='green',
                        opacity=0.7),
            row=1, col=1
        )
        
        # MAPE vs RMSE scatter
        fig.add_trace(
            go.Scatter(x=performance_data['mape'], 
                       y=performance_data['rmse'],
                       mode='markers',
                       name='MAPE vs RMSE',
                       marker=dict(color=performance_data['accuracy'], 
                                 colorscale='Viridis',
                                 showscale=True)),
            row=1, col=2
        )
        
        # Performance by model type (if available)
        if 'model_type' in performance_data.columns:
            fig.add_trace(
                go.Box(x=performance_data['model_type'], 
                       y=performance_data['accuracy'],
                       name='Accuracy by Model'),
                row=2, col=1
            )
        
        # Performance trend over time (if available)
        if 'created_at' in performance_data.columns:
            performance_data_sorted = performance_data.sort_values('created_at')
            fig.add_trace(
                go.Scatter(x=performance_data_sorted['created_at'], 
                           y=performance_data_sorted['accuracy'],
                           mode='lines+markers',
                           name='Accuracy Trend'),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Model Performance Analysis',
            height=800,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating performance metrics chart: {e}")
        return create_empty_chart('Model Performance Metrics')
