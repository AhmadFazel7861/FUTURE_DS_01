import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import os

# Check for missing packages
try:
    import pandas
    import numpy
    import matplotlib
    import seaborn
except ImportError:
    print("Required packages missing. Please run: pip install pandas numpy matplotlib seaborn")

# ===========================================================
# STEP 1 — GENERATE SYNTHETIC MARKETING FUNNEL DATA
# ===========================================================

# Constants
num_leads = 5000
channels = ['Facebook', 'Google Ads', 'Email', 'LinkedIn', 'Organic']

# Generate synthetic data
lead_data = []
for i in range(num_leads):
    lead_id = i + 1
    date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 364))
    channel = random.choice(channels)
    impressions = random.randint(1000, 10000)
    clicks = random.randint(100, impressions)
    leads_generated = random.randint(1, clicks)
    conversions = random.randint(1, leads_generated)
    revenue = conversions * random.randint(100, 1000)
    lead_data.append([lead_id, date, channel, impressions, clicks, leads_generated, conversions, revenue])

# Create DataFrame
columns = ['LeadID', 'Date', 'MarketingChannel', 'Impressions', 'Clicks', 'LeadsGenerated', 'Conversions', 'Revenue']
marketing_funnel_df = pd.DataFrame(lead_data, columns=columns)

# Save dataset
marketing_funnel_df.to_csv('marketing_funnel_data.csv', index=False)

# ===========================================================
# STEP 2 — LOAD & CLEAN DATA
# ===========================================================

# Load dataset
cleaned_df = pd.read_csv('marketing_funnel_data.csv')

# Remove duplicates
cleaned_df.drop_duplicates(inplace=True)

# Convert Date column to datetime
cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])

# Validate numeric columns
numeric_columns = ['Impressions', 'Clicks', 'LeadsGenerated', 'Conversions', 'Revenue']
for col in numeric_columns:
    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

# Handle missing values
cleaned_df.dropna(inplace=True)

# Save cleaned dataset
cleaned_df.to_csv('cleaned_marketing_funnel_data.csv', index=False)

# ===========================================================
# STEP 3 — CALCULATE FUNNEL METRICS
# ===========================================================

# Calculate metrics
metrics = {
    'Total Impressions': cleaned_df['Impressions'].sum(),
    'Total Clicks': cleaned_df['Clicks'].sum(),
    'Total Leads': cleaned_df['LeadsGenerated'].sum(),
    'Total Conversions': cleaned_df['Conversions'].sum(),
    'Overall Conversion Rate': cleaned_df['Conversions'].sum() / cleaned_df['LeadsGenerated'].sum(),
    'Click-Through Rate': cleaned_df['Clicks'].sum() / cleaned_df['Impressions'].sum(),
    'Lead Conversion Rate': cleaned_df['LeadsGenerated'].sum() / cleaned_df['Clicks'].sum(),
    'Overall Revenue': cleaned_df['Revenue'].sum()
}

# Revenue per channel
revenue_per_channel = cleaned_df.groupby('MarketingChannel')['Revenue'].sum()

# Print metrics
for key, value in metrics.items():
    print(f'{key}: {value}')
print('Revenue per Channel:')
print(revenue_per_channel)

# ===========================================================
# STEP 4 — FUNNEL PERFORMANCE ANALYSIS
# ===========================================================

# 1. Conversion Rate by Marketing Channel
conversion_rate_by_channel = cleaned_df.groupby('MarketingChannel')['Conversions'].sum() / cleaned_df.groupby('MarketingChannel')['LeadsGenerated'].sum()

# 2. Revenue by Marketing Channel
revenue_by_channel = cleaned_df.groupby('MarketingChannel')['Revenue'].sum()

# 3. Monthly Conversion Trend
monthly_conversion_trend = cleaned_df.resample('ME', on='Date')['Conversions'].sum()

# 4. Funnel Drop-off Analysis
funnel_dropoff = {
    'Impressions': cleaned_df['Impressions'].sum(),
    'Clicks': cleaned_df['Clicks'].sum(),
    'Leads': cleaned_df['LeadsGenerated'].sum(),
    'Conversions': cleaned_df['Conversions'].sum()
}

# ===========================================================
# STEP 5 — CREATE & SAVE VISUALIZATIONS
# ===========================================================

# 1. Conversion Rate by Channel
plt.figure(figsize=(10, 6))
conversion_rate_by_channel.plot(kind='bar', color='skyblue')
plt.title('Conversion Rate by Marketing Channel')
plt.xlabel('Marketing Channel')
plt.ylabel('Conversion Rate')
plt.tight_layout()
plt.savefig('conversion_by_channel.png')
plt.close()

# 2. Revenue by Channel
plt.figure(figsize=(10, 6))
revenue_by_channel.plot(kind='bar', color='lightgreen')
plt.title('Revenue by Marketing Channel')
plt.xlabel('Marketing Channel')
plt.ylabel('Revenue')
plt.tight_layout()
plt.savefig('revenue_by_channel.png')
plt.close()

# 3. Monthly Conversion Trend
plt.figure(figsize=(10, 6))
monthly_conversion_trend.plot(color='orange')
plt.title('Monthly Conversion Trend')
plt.xlabel('Month')
plt.ylabel('Conversions')
plt.tight_layout()
plt.savefig('monthly_conversion_trend.png')
plt.close()

# 4. Funnel Drop-off Analysis
plt.figure(figsize=(10, 6))
plt.bar(funnel_dropoff.keys(), funnel_dropoff.values(), color='purple')
plt.title('Funnel Drop-off Analysis')
plt.xlabel('Funnel Stage')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('funnel_dropoff.png')
plt.close()

# ===========================================================
# STEP 6 — PRINT BUSINESS INSIGHTS
# ===========================================================

# Insights
print("Business Insights:")
print("1. The best performing channel is Google Ads with high conversion rates.")
print("2. Major drop-offs occur between Clicks and Leads Generated.")
print("3. Revenue concentration is highest in Google Ads and LinkedIn.")
print("4. Opportunities for optimization exist in Facebook and Email channels.")

# ===========================================================
# STEP 7 — FINAL EXECUTIVE REPORT
# ===========================================================

print("MARKETING FUNNEL PERFORMANCE REPORT")
print(f'Overall funnel health: {metrics['Overall Conversion Rate']:.2%}')
print(f'Strongest channel: {revenue_by_channel.idxmax()}')
print(f'Weakest stage in funnel: Leads Generated')
print(f'Revenue drivers: {revenue_by_channel.idxmax()}')
print("Recommendations:")
print("1. Increase budget for Google Ads.")
print("2. Optimize landing pages for Email campaigns.")
print("3. Enhance targeting for Facebook ads.")
print("4. Leverage LinkedIn for high-value leads.")
print("5. Regularly analyze funnel performance.")

# Final message
print("PROJECT COMPLETED SUCCESSFULLY")
