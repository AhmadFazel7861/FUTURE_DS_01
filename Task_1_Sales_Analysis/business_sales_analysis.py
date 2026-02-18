#!/usr/bin/env python3
"""
Business Sales Performance Analysis

This script creates a synthetic sales dataset, cleans it, computes key
performance metrics, produces a few standard analyses and charts, and
saves the cleaned data and image files for review.
"""

# --------------------------------------------------
# Imports and dependency check
# If required packages are missing, the script will prompt to install them
# --------------------------------------------------
import sys
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime, timedelta
    import random
    import os
except Exception as e:
    print("Missing required libraries. Please install with:")
    print("pip install pandas numpy matplotlib seaborn")
    sys.exit(1)


# ---------------------------
# File and runtime configuration
# ---------------------------
OUTPUT_DIR = os.getcwd()
SALES_CSV = os.path.join(OUTPUT_DIR, "sales.csv")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_sales_data.csv")
PNG_MONTHLY = os.path.join(OUTPUT_DIR, "monthly_revenue.png")
PNG_TOP = os.path.join(OUTPUT_DIR, "top_products.png")
PNG_CATEGORY = os.path.join(OUTPUT_DIR, "category_revenue.png")
PNG_REGION = os.path.join(OUTPUT_DIR, "region_revenue.png")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------
# Generate a synthetic sales dataset
# ---------------------------
def generate_synthetic_sales(n_rows=1500, out_path=SALES_CSV):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_range = (end_date - start_date).days + 1

    # Define a set of products and their categories. This provides
    # variety across categories and products for analysis.
    products = [
        ("AlphaPhone", "Electronics"),
        ("BetaLaptop", "Electronics"),
        ("GammaHeadset", "Accessories"),
        ("DeltaCharger", "Accessories"),
        ("OmegaChair", "Furniture"),
        ("SigmaDesk", "Furniture"),
        ("ZetaNotebook", "Office Supplies"),
        ("ThetaPen", "Office Supplies"),
        ("IotaLamp", "Home"),
        ("KappaMug", "Home"),
    ]

    regions = ["North", "South", "East", "West"]

    rows = []
    for i in range(1, n_rows + 1):
        order_id = f"ORD{i:06d}"
        # Random date
        rand_days = random.randrange(days_range)
        order_date = (start_date + timedelta(days=rand_days)).strftime("%Y-%m-%d")

        product, category = random.choice(products)
        region = random.choice(regions)
        quantity = random.randint(1, 10)
        # Price skewed: many mid-range prices with some variation
        price = round(np.random.uniform(10, 500), 2)
        revenue = round(quantity * price, 2)

        rows.append({
            "OrderID": order_id,
            "OrderDate": order_date,
            "ProductName": product,
            "Category": category,
            "Region": region,
            "Quantity": quantity,
            "Price": price,
            "Revenue": revenue,
        })

    df = pd.DataFrame(rows)
    # Add a few duplicate rows and some missing values to mimic messy data
    dup_indices = np.random.choice(df.index, size=3, replace=False)
    df = pd.concat([df, df.loc[dup_indices]], ignore_index=True).reset_index(drop=True)

    # Introduce a few missing values
    for col in ["Price", "Quantity", "Region"]:
        for idx in np.random.choice(df.index, size=3, replace=False):
            df.at[idx, col] = np.nan

    df.to_csv(out_path, index=False)
    return df


# ---------------------------
# Load the dataset and perform cleaning
# - drop duplicates
# - fill or coerce missing values
# - ensure correct dtypes and a Month column for grouping
# - write out a cleaned CSV
# ---------------------------
def load_and_clean(path=SALES_CSV):
    df = pd.read_csv(path)

    # drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    # Fill or impute simple missing values
    # Price: use median as a neutral estimate
    if df['Price'].isnull().any():
        median_price = df['Price'].median()
        df['Price'] = df['Price'].fillna(median_price)
    # Quantity: default to 1 when missing
    if df['Quantity'].isnull().any():
        df['Quantity'] = df['Quantity'].fillna(1).astype(int)
    # Region: mark as Unknown so it's still groupable
    if df['Region'].isnull().any():
        df['Region'] = df['Region'].fillna('Unknown')

    # Normalize types and recalc revenue to avoid inconsistencies
    df['Quantity'] = df['Quantity'].astype(int)
    df['Price'] = df['Price'].astype(float)
    df['Revenue'] = (df['Quantity'] * df['Price']).round(2)

    # Parse dates and create a Month column for aggregates
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['OrderDate']).reset_index(drop=True)
    df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)

    # persist cleaned data
    df.to_csv(CLEANED_CSV, index=False)

    return df, before, after


# ---------------------------
# Compute and print key performance metrics
# ---------------------------
def calculate_kpis(df):
    total_revenue = df['Revenue'].sum()
    total_orders = df['OrderID'].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0
    total_quantity = df['Quantity'].sum()

    print("\n" + "=" * 60)
    print("KEY PERFORMANCE INDICATORS")
    print("=" * 60)
    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Total Orders: {total_orders:,}")
    print(f"Average Order Value: ${avg_order_value:,.2f}")
    print(f"Total Quantity Sold: {total_quantity:,}")
    print("=" * 60 + "\n")

    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'avg_order_value': avg_order_value,
        'total_quantity': total_quantity,
    }


# ---------------------------
# Business analysis: monthly trend, top products, categories, regions
# Each block prints results and short human-readable insights
# ---------------------------
def business_analysis(df):
    # 1. Monthly Revenue Trend
    monthly = df.groupby('Month', sort=False)['Revenue'].sum().reset_index()

    # 2. Top 10 Products by Revenue
    top_products = df.groupby('ProductName')['Revenue'].sum().reset_index()
    top_products = top_products.sort_values('Revenue', ascending=False).head(10)

    # 3. Revenue by Category
    category_rev = df.groupby('Category')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)

    # 4. Revenue by Region
    region_rev = df.groupby('Region')['Revenue'].sum().reset_index().sort_values('Revenue', ascending=False)

    # Print insights after each analysis
    print("MONTHLY REVENUE TREND")
    print(monthly.to_string(index=False))
    print_insights_monthly(monthly)

    print("\nTOP 10 PRODUCTS BY REVENUE")
    print(top_products.to_string(index=False))
    print_insights_products(top_products)

    print("\nREVENUE BY CATEGORY")
    print(category_rev.to_string(index=False))
    print_insights_category(category_rev)

    print("\nREVENUE BY REGION")
    print(region_rev.to_string(index=False))
    print_insights_region(region_rev)

    return monthly, top_products, category_rev, region_rev


# ---------------------------
# Create and save charts used in the analysis
# The charts are saved to PNG files in the working directory
# ---------------------------
def create_visualizations(monthly, top_products, category_rev, region_rev):
    sns.set(style='whitegrid')

    # Monthly Revenue Line Chart
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x='Month', y='Revenue', marker='o')
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Month')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PNG_MONTHLY, dpi=300)
    plt.close()

    # Top Products Bar Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_products, x='Revenue', y='ProductName', palette='viridis')
    plt.title('Top 10 Products by Revenue')
    plt.xlabel('Revenue')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.savefig(PNG_TOP, dpi=300)
    plt.close()

    # Category Revenue Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=category_rev, x='Category', y='Revenue', palette='coolwarm')
    plt.title('Revenue by Category')
    plt.xlabel('Category')
    plt.ylabel('Revenue')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(PNG_CATEGORY, dpi=300)
    plt.close()

    # Region Revenue Bar Chart
    plt.figure(figsize=(8, 6))
    sns.barplot(data=region_rev, x='Region', y='Revenue', palette='magma')
    plt.title('Revenue by Region')
    plt.xlabel('Region')
    plt.ylabel('Revenue')
    plt.tight_layout()
    plt.savefig(PNG_REGION, dpi=300)
    plt.close()


# ---------------------------
# Short helper functions that print concise business insights
# These are intentionally human-friendly and actionable
# ---------------------------
def print_insights_monthly(monthly):
    # Convert Month to datetime for sorting and insight
    try:
        m = monthly.copy()
        m['Month_dt'] = pd.to_datetime(m['Month'])
        peak = m.loc[m['Revenue'].idxmax()]
        low = m.loc[m['Revenue'].idxmin()]
        print("\nInsights:")
        print(f"- Peak month: {peak['Month']} with revenue ${peak['Revenue']:,.2f}.")
        print(f"- Lowest month: {low['Month']} with revenue ${low['Revenue']:,.2f}.")
        print("- Trend: examine seasonal campaigns or inventory around peak months.")
    except Exception:
        print("- Monthly trend summary not available due to data format.")


def print_insights_products(top_products):
    print("\nInsights:")
    if not top_products.empty:
        best = top_products.iloc[0]
        print(f"- Best product: {best['ProductName']} generating ${best['Revenue']:,.2f}.")
        print("- Focus marketing and inventory on top performers; consider bundling.")
    else:
        print("- No product data available.")


def print_insights_category(category_rev):
    print("\nInsights:")
    if not category_rev.empty:
        top = category_rev.iloc[0]
        print(f"- Leading category: {top['Category']} with ${top['Revenue']:,.2f} revenue.")
        print("- Consider category-specific promotions and margin improvement.")
    else:
        print("- No category data available.")


def print_insights_region(region_rev):
    print("\nInsights:")
    if not region_rev.empty:
        top = region_rev.iloc[0]
        print(f"- Top region: {top['Region']} generating ${top['Revenue']:,.2f}.")
        print("- Tailor logistics and local campaigns to high-performing regions.")
    else:
        print("- No region data available.")


# ---------------------------
# Final summary and recommendations
# ---------------------------
def final_summary(df, monthly, top_products, category_rev, region_rev):
    total_revenue = df['Revenue'].sum()
    best_region = region_rev.iloc[0] if not region_rev.empty else None
    best_product = top_products.iloc[0] if not top_products.empty else None
    best_month_row = monthly.loc[monthly['Revenue'].idxmax()] if not monthly.empty else None

    print("\n" + "#" * 60)
    print("BUSINESS SUMMARY REPORT")
    print("#" * 60)
    print(f"Overall Performance: Total Revenue = ${total_revenue:,.2f} over {df['OrderID'].nunique():,} orders.")
    if best_region is not None:
        print(f"Best Performing Region: {best_region['Region']} (${best_region['Revenue']:,.2f})")
    if best_product is not None:
        print(f"Best Product: {best_product['ProductName']} (${best_product['Revenue']:,.2f})")
    if best_month_row is not None:
        print(f"Best Month: {best_month_row['Month']} (${best_month_row['Revenue']:,.2f})")

    print("\nTop Recommendations:")
    print("1. Prioritize inventory and promotional spend for top products and peak months to maximize revenue.")
    print("2. Invest in regional campaigns and distribution for the best-performing region to capture more market share.")
    print("3. Analyze underperforming months/categories to identify causes (pricing, stockouts, seasonality) and run targeted interventions.")
    print("#" * 60 + "\n")


# ---------------------------
# Main entry point: runs the full pipeline end-to-end
# ---------------------------
def main():
    print("Starting Business Sales Performance Analysis...")

    # Generate dataset if not present
    if not os.path.exists(SALES_CSV):
        print(f"Generating synthetic dataset and saving to {SALES_CSV}...")
        generate_synthetic_sales(n_rows=1500, out_path=SALES_CSV)
    else:
        print(f"Found existing {SALES_CSV}; will use it.")

    # Load and clean
    df_clean, before_dup_count, after_dup_count = load_and_clean(SALES_CSV)

    # KPIs
    kpis = calculate_kpis(df_clean)

    # Business analysis
    monthly, top_products, category_rev, region_rev = business_analysis(df_clean)

    # Visualizations
    print("\nCreating and saving visualizations...")
    create_visualizations(monthly, top_products, category_rev, region_rev)
    print(f"Charts saved: {PNG_MONTHLY}, {PNG_TOP}, {PNG_CATEGORY}, {PNG_REGION}")

    # Final summary
    final_summary(df_clean, monthly, top_products, category_rev, region_rev)

    # Cleaned data already saved in load_and_clean
    print(f"Cleaned data saved to: {CLEANED_CSV}")

    print("PROJECT COMPLETED SUCCESSFULLY")


if __name__ == '__main__':
    main()
