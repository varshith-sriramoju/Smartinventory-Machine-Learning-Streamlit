import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.database import DatabaseManager
from utils.data_processing import DataProcessor

st.set_page_config(
    page_title="Data Upload - SmartInventory",
    page_icon="📁",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize database and data processor"""
    db = DatabaseManager()
    processor = DataProcessor()
    return db, processor

def main():
    st.title("📁 Data Upload")
    st.markdown("Upload your historical sales data to begin forecasting")

    db, processor = init_components()

    # File upload section
    st.header("1. Upload Sales Data")

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your historical sales data. Required columns: date, product (name/id), sales quantity. Optional: price, category, store_id"
    )

    if uploaded_file is not None:
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ File uploaded successfully! Found {len(df)} records with {len(df.columns)} columns.")

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                if date_cols:
                    try:
                        date_series = pd.to_datetime(df[date_cols[0]], errors='coerce')
                        valid_dates = date_series.dropna()
                        if len(valid_dates) > 0:
                            date_span = (valid_dates.max() - valid_dates.min()).days
                            st.metric("Date Range (days)", f"{date_span:,}")
                        else:
                            st.metric("Date Range", "Invalid dates")
                    except:
                        st.metric("Date Range", "Parse error")
                else:
                    st.metric("Date Range", "No date column")

            st.divider()

            # Data validation
            st.header("2. Data Validation")

            validation_results = processor.validate_data(df)

            if not validation_results['is_valid']:
                st.error("⚠️ Data validation issues found:")
                for error in validation_results['errors']:
                    st.error(f"• {error}")

                st.markdown("**Please fix these issues and re-upload your file.**")
                return

            st.success("✅ Data structure is valid!")

            # Show data preview
            st.header("3. Data Preview")

            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Sample Values': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
            })
            st.dataframe(col_info, width="stretch")

            # Data sample
            st.subheader("Data Sample (First 10 rows)")
            st.dataframe(df.head(10), width="stretch")

            # Column mapping
            st.header("4. Column Mapping")
            st.markdown("Map your columns to the required fields:")

            col1, col2 = st.columns(2)

            with col1:
                date_col = st.selectbox("Date Column *", df.columns,
                                      index=next((i for i, col in enumerate(df.columns) if 'date' in col.lower()), 0))
                product_col = st.selectbox("Product Name/ID Column *", df.columns,
                                         index=next((i for i, col in enumerate(df.columns) if any(keyword in col.lower() for keyword in ['product', 'item', 'name'])), 0))
                quantity_col = st.selectbox("Sales Quantity Column *", df.columns,
                                          index=next((i for i, col in enumerate(df.columns) if any(keyword in col.lower() for keyword in ['quantity', 'qty', 'sales', 'units'])), 0))

            with col2:
                price_col = st.selectbox("Price Column (Optional)", [None] + list(df.columns),
                                       index=next((i+1 for i, col in enumerate(df.columns) if 'price' in col.lower()), 0))
                category_col = st.selectbox("Category Column (Optional)", [None] + list(df.columns),
                                          index=next((i+1 for i, col in enumerate(df.columns) if 'category' in col.lower()), 0))
                store_col = st.selectbox("Store ID Column (Optional)", [None] + list(df.columns),
                                       index=next((i+1 for i, col in enumerate(df.columns) if 'store' in col.lower()), 0))

            # Data processing options
            st.header("5. Data Processing Options")

            col1, col2 = st.columns(2)

            with col1:
                handle_duplicates = st.checkbox("Remove duplicate records", value=True)
                handle_missing = st.selectbox(
                    "Handle missing values",
                    ["Keep as is", "Remove records with missing values", "Fill with defaults"]
                )

            with col2:
                date_format = st.selectbox(
                    "Date format",
                    ["Auto-detect", "YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY", "YYYY/MM/DD"]
                )
                aggregate_duplicates = st.checkbox("Aggregate duplicate date-product combinations", value=True)

            # Process and save data
            if st.button("🚀 Process and Save Data", type="primary"):
                with st.spinner("Processing data..."):
                    try:
                        # Process the data
                        processed_df = processor.process_uploaded_data(
                            df, date_col, product_col, quantity_col,
                            price_col, category_col, store_col,
                            handle_duplicates, handle_missing, date_format, aggregate_duplicates
                        )

                        if processed_df is None or processed_df.empty:
                            st.error("❌ Data processing failed. Please check your data and try again.")
                            return

                        # Save to database
                        success = db.save_sales_data(processed_df)

                        if success:
                            st.success("✅ Data processed and saved successfully!")
                            st.balloons()

                            # Show final statistics
                            st.subheader("📊 Final Dataset Statistics")
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Final Record Count", f"{len(processed_df):,}")

                            with col2:
                                unique_products = processed_df['product_name'].nunique()
                                st.metric("Unique Products", f"{unique_products:,}")

                            with col3:
                                date_range = (processed_df['date'].max() - processed_df['date'].min()).days
                                st.metric("Date Range (days)", f"{date_range:,}")

                            with col4:
                                total_sales = processed_df['sales_quantity'].sum()
                                st.metric("Total Units Sold", f"{total_sales:,.0f}")

                            st.info("🎉 You can now proceed to **Data Exploration** to analyze your data!")

                        else:
                            st.error("❌ Failed to save data to database. Please check your database connection.")

                    except Exception as e:
                        st.error(f"❌ Error processing data: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.markdown("Please ensure your file is a valid CSV or Excel file with proper formatting.")

    # Existing data section
    st.divider()
    st.header("📊 Current Database Status")

    try:
        existing_data_count = db.get_total_records()
        if existing_data_count > 0:
            st.success(f"Database contains {existing_data_count:,} records")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📋 View Sample Data"):
                    sample_data = db.get_sample_data(100)
                    if sample_data is not None and not sample_data.empty:
                        st.dataframe(sample_data, width="stretch")
                    else:
                        st.warning("No sample data available")

            with col2:
                if st.button("🗑️ Clear All Data", type="secondary"):
                    if 'confirm_delete' not in st.session_state:
                        st.session_state.confirm_delete = False

                    if st.session_state.confirm_delete:
                        db.clear_all_data()
                        st.session_state.confirm_delete = False
                        st.success("All data cleared successfully!")
                        st.rerun()
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ Click again to confirm data deletion")
        else:
            st.info("Database is empty. Upload a file to get started.")

    except Exception as e:
        st.error(f"Error checking database status: {str(e)}")

if __name__ == "__main__":
    main()
