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
from utils.inventory_optimization import InventoryOptimizer

st.set_page_config(
    page_title="Inventory Management - SmartInventory",
    page_icon="📦",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize components"""
    db = DatabaseManager()
    inventory_optimizer = InventoryOptimizer()
    return db, inventory_optimizer

def main():
    st.title("📦 Inventory Management")
    st.markdown("Optimize inventory levels with AI-powered recommendations")
    
    db, inventory_optimizer = init_components()
    
    # Check if forecasts exist
    forecasts = db.get_recent_forecasts()
    
    if forecasts is None or forecasts.empty:
        st.warning("🔮 No forecasts available. Please generate forecasts first using the **Forecasting** page.")
        
        st.subheader("Getting Started")
        st.markdown("""
        1. Navigate to the **Forecasting** page
        2. Generate forecasts for your products
        3. Return here to get inventory recommendations
        """)
        return
    
    st.success(f"✅ Found forecasts for {forecasts['product_name'].nunique()} products")
    
    # Inventory configuration
    st.header("⚙️ Inventory Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        service_level = st.slider(
            "Service Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            help="Desired probability of not having a stockout"
        )
    
    with col2:
        lead_time = st.number_input(
            "Lead Time (days)",
            min_value=1,
            max_value=60,
            value=7,
            help="Time between placing and receiving an order"
        )
    
    with col3:
        safety_stock_factor = st.slider(
            "Safety Stock Factor",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Multiplier for safety stock calculation"
        )
    
    with col4:
        budget_constraint = st.number_input(
            "Budget Constraint ($)",
            min_value=0,
            value=0,
            help="Maximum budget for inventory purchases (0 = no limit)"
        )
    
    # Generate recommendations
    if st.button("🚀 Generate Inventory Recommendations", type="primary", use_container_width=True):
        
        with st.spinner("Calculating optimal inventory levels..."):
            
            # Get current inventory levels (simulated for demo)
            current_inventory = db.get_current_inventory()
            
            if current_inventory.empty:
                st.warning("No current inventory data available. Using estimated values.")
                # Create simulated current inventory
                products = forecasts['product_name'].unique()
                current_inventory = pd.DataFrame({
                    'product_name': products,
                    'current_stock': np.random.randint(10, 200, len(products))
                })
            
            # Calculate recommendations
            recommendations = inventory_optimizer.calculate_recommendations(
                forecasts, current_inventory, service_level, lead_time, safety_stock_factor
            )
            
            if recommendations is None or recommendations.empty:
                st.error("❌ Could not generate recommendations. Please check your data.")
                return
            
            # Apply budget constraint if specified
            if budget_constraint > 0:
                recommendations = inventory_optimizer.optimize_inventory_levels(
                    recommendations, budget_constraint
                )
            
            # Save recommendations to database
            db.save_inventory_recommendations(recommendations)
            
            st.success(f"✅ Generated recommendations for {len(recommendations)} products!")
            
            # Display recommendations
            st.header("📋 Inventory Recommendations")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            reorder_needed = len(recommendations[recommendations['action'] == 'reorder'])
            overstock_items = len(recommendations[recommendations['action'] == 'reduce'])
            total_investment = recommendations[recommendations['action'] == 'reorder']['recommended_order_quantity'].sum()
            potential_savings = recommendations[recommendations['action'] == 'reduce']['excess_inventory'].sum()
            
            with col1:
                st.metric("🔄 Items to Reorder", reorder_needed)
            with col2:
                st.metric("📈 Overstock Items", overstock_items)
            with col3:
                st.metric("💰 Investment Needed", f"{total_investment:,.0f} units")
            with col4:
                st.metric("💡 Potential Savings", f"{potential_savings:,.0f} units")
            
            st.divider()
            
            # Action-based recommendations
            tabs = st.tabs(["🚨 Priority Actions", "📊 All Recommendations", "📈 Visualizations"])
            
            with tabs[0]:
                # Priority actions
                st.subheader("🔴 Critical - Immediate Reorder Required")
                
                critical_items = recommendations[
                    (recommendations['action'] == 'reorder') & 
                    (recommendations['urgency_score'] >= 80)
                ].sort_values('urgency_score', ascending=False)
                
                if not critical_items.empty:
                    st.dataframe(
                        critical_items[['product_name', 'current_stock', 'safety_stock', 'recommended_order_quantity', 'urgency_score']],
                        column_config={
                            'product_name': 'Product',
                            'current_stock': st.column_config.NumberColumn('Current Stock', format="%.0f"),
                            'safety_stock': st.column_config.NumberColumn('Safety Stock', format="%.0f"),
                            'recommended_order_quantity': st.column_config.NumberColumn('Order Quantity', format="%.0f"),
                            'urgency_score': st.column_config.ProgressColumn('Urgency', min_value=0, max_value=100)
                        },
                        use_container_width=True
                    )
                else:
                    st.success("✅ No critical stock issues found!")
                
                st.subheader("🟡 Medium Priority - Reorder Soon")
                
                medium_items = recommendations[
                    (recommendations['action'] == 'reorder') & 
                    (recommendations['urgency_score'] >= 40) &
                    (recommendations['urgency_score'] < 80)
                ].sort_values('urgency_score', ascending=False)
                
                if not medium_items.empty:
                    st.dataframe(
                        medium_items[['product_name', 'current_stock', 'safety_stock', 'recommended_order_quantity', 'urgency_score']],
                        column_config={
                            'product_name': 'Product',
                            'current_stock': st.column_config.NumberColumn('Current Stock', format="%.0f"),
                            'safety_stock': st.column_config.NumberColumn('Safety Stock', format="%.0f"),
                            'recommended_order_quantity': st.column_config.NumberColumn('Order Quantity', format="%.0f"),
                            'urgency_score': st.column_config.ProgressColumn('Urgency', min_value=0, max_value=100)
                        },
                        use_container_width=True
                    )
                else:
                    st.info("No medium priority items found.")
                
                st.subheader("🔵 Overstock - Consider Reduction")
                
                overstock_items = recommendations[
                    recommendations['action'] == 'reduce'
                ].sort_values('excess_inventory', ascending=False)
                
                if not overstock_items.empty:
                    st.dataframe(
                        overstock_items[['product_name', 'current_stock', 'safety_stock', 'excess_inventory']],
                        column_config={
                            'product_name': 'Product',
                            'current_stock': st.column_config.NumberColumn('Current Stock', format="%.0f"),
                            'safety_stock': st.column_config.NumberColumn('Optimal Stock', format="%.0f"),
                            'excess_inventory': st.column_config.NumberColumn('Excess Units', format="%.0f")
                        },
                        use_container_width=True
                    )
                else:
                    st.success("✅ No overstock issues found!")
            
            with tabs[1]:
                # All recommendations
                st.subheader("📊 Complete Inventory Analysis")
                
                # Add action filter
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    action_filter = st.selectbox(
                        "Filter by Action",
                        ["All", "reorder", "reduce", "maintain"]
                    )
                
                filtered_recommendations = recommendations.copy()
                if action_filter != "All":
                    filtered_recommendations = filtered_recommendations[
                        filtered_recommendations['action'] == action_filter
                    ]
                
                # Display table with all columns
                st.dataframe(
                    filtered_recommendations,
                    column_config={
                        'product_name': 'Product',
                        'current_stock': st.column_config.NumberColumn('Current Stock', format="%.0f"),
                        'forecasted_demand': st.column_config.NumberColumn('Daily Demand', format="%.2f"),
                        'safety_stock': st.column_config.NumberColumn('Safety Stock', format="%.0f"),
                        'reorder_point': st.column_config.NumberColumn('Reorder Point', format="%.0f"),
                        'eoq': st.column_config.NumberColumn('EOQ', format="%.0f"),
                        'recommended_order_quantity': st.column_config.NumberColumn('Order Qty', format="%.0f"),
                        'action': 'Action',
                        'urgency_score': st.column_config.ProgressColumn('Urgency', min_value=0, max_value=100),
                        'excess_inventory': st.column_config.NumberColumn('Excess', format="%.0f")
                    },
                    use_container_width=True,
                    height=400
                )
            
            with tabs[2]:
                # Visualizations
                st.subheader("📈 Inventory Analysis Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Current vs Recommended Stock Levels
                    top_products = recommendations.head(10)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name='Current Stock',
                        x=top_products['product_name'],
                        y=top_products['current_stock'],
                        marker_color='lightblue'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Safety Stock',
                        x=top_products['product_name'],
                        y=top_products['safety_stock'],
                        marker_color='orange'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Reorder Point',
                        x=top_products['product_name'],
                        y=top_products['reorder_point'],
                        marker_color='red'
                    ))
                    
                    fig.update_layout(
                        title='Current vs Optimal Stock Levels (Top 10 Products)',
                        xaxis_title='Product',
                        yaxis_title='Quantity',
                        barmode='group',
                        height=400
                    )
                    fig.update_xaxes(tickangle=45)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Action Distribution
                    action_counts = recommendations['action'].value_counts()
                    
                    fig = px.pie(
                        values=action_counts.values,
                        names=action_counts.index,
                        title='Distribution of Recommended Actions',
                        color_discrete_map={
                            'reorder': '#ff7f0e',
                            'reduce': '#1f77b4',
                            'maintain': '#2ca02c'
                        }
                    )
                    fig.update_layout(height=400)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Urgency Score Distribution
                fig = px.histogram(
                    recommendations,
                    x='urgency_score',
                    nbins=20,
                    title='Distribution of Urgency Scores',
                    labels={'urgency_score': 'Urgency Score', 'count': 'Number of Products'}
                )
                fig.add_vline(x=80, line_dash="dash", line_color="red", 
                             annotation_text="Critical Threshold")
                fig.add_vline(x=40, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Priority Threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Export recommendations
            st.divider()
            st.header("📥 Export Recommendations")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = recommendations.to_csv(index=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📊 Download All Recommendations",
                    data=csv,
                    file_name=f"inventory_recommendations_{timestamp}.csv",
                    mime="text/csv",
                    type="secondary"
                )
            
            with col2:
                # Export only reorder items
                reorder_items = recommendations[recommendations['action'] == 'reorder']
                if not reorder_items.empty:
                    csv = reorder_items.to_csv(index=False)
                    st.download_button(
                        label="🔄 Download Reorder List",
                        data=csv,
                        file_name=f"reorder_list_{timestamp}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                else:
                    st.info("No items need reordering")
            
            with col3:
                # Export overstock items
                overstock_items = recommendations[recommendations['action'] == 'reduce']
                if not overstock_items.empty:
                    csv = overstock_items.to_csv(index=False)
                    st.download_button(
                        label="📉 Download Overstock List",
                        data=csv,
                        file_name=f"overstock_list_{timestamp}.csv",
                        mime="text/csv",
                        type="secondary"
                    )
                else:
                    st.info("No overstock items found")

if __name__ == "__main__":
    main()
