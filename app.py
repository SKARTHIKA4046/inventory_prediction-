import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Dairy Sales Dashboard", layout="wide")
st.title("🥛 Dairy Sales Analytics & Inventory Prediction")

uploaded_file = st.file_uploader("Upload Dairy Sales CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # ----------------------------
    # SMART COLUMN DETECTION
    # ----------------------------
    def find_column(possible_names):
        for col in df.columns:
            for name in possible_names:
                if name.lower() in col.lower():
                    return col
        return None

    product_name_col = find_column(["productname", "product_name", "product name"])
    product_id_col = find_column(["productid", "product_id"])

    if product_name_col:
        product_col = product_name_col
    else:
        product_col = find_column(["product", "item"])

    location_col = find_column(["location", "country", "region", "state"])
    date_col = find_column(["date"])
    quantity_col = find_column(["quantity", "qty", "unit", "sales", "sold"])

    if not product_col or not location_col or not date_col or not quantity_col:
        st.error("Required columns not found in dataset.")
        st.stop()

    # Rename columns internally
    df = df.rename(columns={
        product_col: "Product",
        location_col: "Location",
        date_col: "Date",
        quantity_col: "Quantity"
    })

    if product_name_col and product_id_col:
        df["Product"] = df[product_name_col].astype(str)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Date", "Quantity"])
    df = df.sort_values("Date")

    # ----------------------------
    # SIDEBAR FILTERS
    # ----------------------------
    st.sidebar.header("🔎 Filters")

    selected_product = st.sidebar.multiselect(
        "Select Product",
        sorted(df["Product"].unique()),
        default=sorted(df["Product"].unique())
    )

    selected_location = st.sidebar.multiselect(
        "Select Location",
        sorted(df["Location"].unique()),
        default=sorted(df["Location"].unique())
    )

    min_date = df["Date"].min()
    max_date = df["Date"].max()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date]
    )

    filtered_df = df[
        (df["Product"].isin(selected_product)) &
        (df["Location"].isin(selected_location)) &
        (df["Date"] >= pd.to_datetime(date_range[0])) &
        (df["Date"] <= pd.to_datetime(date_range[1]))
    ]

    if filtered_df.empty:
        st.warning("No data available after applying filters.")
        st.stop()

    # ----------------------------
    # SALES TREND LINE CHART
    # ----------------------------
    st.subheader("📈 Sales Trend Over Time")

    daily_sales = (
        filtered_df.groupby("Date")["Quantity"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    fig1, ax1 = plt.subplots()
    ax1.plot(daily_sales["Date"], daily_sales["Quantity"], color="#1f77b4", linewidth=2)  # Blue line
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Quantity Sold")
    ax1.set_title("Daily Sales Trend", color="#2c3e50")
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # ----------------------------
    # BAR CHART – PRODUCT DEMAND
    # ----------------------------
    st.subheader("📊 Total Demand by Product")

    product_sales = (
        filtered_df.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    colors = plt.cm.tab20.colors  # Use same 20-color palette as pie chart
    fig2, ax2 = plt.subplots()
    product_sales.plot(kind="bar", ax=ax2, color=colors[:len(product_sales)])  # Match number of bars
    ax2.set_xlabel("Product")
    ax2.set_ylabel("Total Quantity Sold")
    ax2.set_title("Demand by Product", color="#2c3e50")
    st.pyplot(fig2)

    # ----------------------------
    # PIE CHART – PRODUCT SHARE
    # ----------------------------
    st.subheader("🥧 Product Demand Distribution")

    fig3, ax3 = plt.subplots()
    ax3.pie(product_sales, labels=product_sales.index, autopct="%1.1f%%", colors=colors[:len(product_sales)])
    ax3.set_title("Demand Share", color="#2c3e50")
    st.pyplot(fig3)

    # ----------------------------
    # 30-DAY PREDICTION
    # ----------------------------
    st.subheader("🔮 30-Day Sales Prediction")

    daily_sales["Day_Number"] = np.arange(len(daily_sales))

    X = daily_sales[["Day_Number"]]
    y = daily_sales["Quantity"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(len(daily_sales), len(daily_sales) + 30).reshape(-1, 1)
    future_predictions = model.predict(future_days)

    future_dates = pd.date_range(
        start=daily_sales["Date"].max() + pd.Timedelta(days=1),
        periods=30,
        freq="D"
    )

    fig4, ax4 = plt.subplots()
    ax4.plot(daily_sales["Date"], y, label="Actual Sales", color="#1f77b4", linewidth=2)
    ax4.plot(future_dates, future_predictions, linestyle="--", label="Predicted Sales", color="#ff7f0e", linewidth=2)
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Quantity")
    ax4.set_title("Next 30 Days Forecast", color="#2c3e50")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig4)

    # ----------------------------
    # INVENTORY SUGGESTION
    # ----------------------------
    st.subheader("📦 Inventory Suggestion")

    avg_predicted_daily = np.mean(future_predictions)
    suggested_inventory = int(avg_predicted_daily * 30)
    safety_stock = int(suggested_inventory * 0.10)
    total_with_safety = suggested_inventory + safety_stock

    col1, col2 = st.columns(2)
    col1.metric("Suggested Inventory (Next 30 Days)", suggested_inventory, delta_color="normal")
    col2.metric("With 10% Safety Stock", total_with_safety, delta_color="off")

else:
    st.info("Upload your dairy sales dataset to begin.")
