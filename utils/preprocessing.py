import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
df=pd.read_csv("/content/enterprise_multiagent_decision_support (1).csv")


#timestamp
# Convert text dates into actual datetime objects
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
#missing values
column=df.select_dtypes(include=["float64","int64"]).columns
for i in column:
  if df[i].isnull().sum()>0:
    df[i]=df[i].fillna(df[i].median())
    categorical_cols=df.select_dtypes(include=['object','bool']).columns
for col in categorical_cols:
      if df[col].isnull().sum()>0:
        df[col]= df[col].fillna("not provided")
print("Remaining Missing values:")
print(df.isnull().sum().max())
# 1. Define the numerical columns that are prone to extreme values
target_outlier_cols = [
    'Transaction_Amount_USD',
    'Revenue_Impact_USD',
    'Cost_Savings_Potential_USD',
    'Input_Tokens',
    'Output_Tokens'
]

# 2. Cap the outliers using the standard IQR mathematical formula
for col in target_outlier_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # np.clip forces any value outside the bounds to equal the boundary limit
        df[col] = np.clip(df[col], lower_bound, upper_bound)

print("Outliers capped successfully!")
# Save the pristine dataset to a new CSV file
output_filename = 'cleaned_enterprise_data_final.csv'
df.to_csv(output_filename, index=False)

print(f"Success! Your clean data is saved as: {output_filename}")
df.info()
# 1. Group the data by Department and calculate total Revenue Impact
revenue_by_dept = df.groupby('Department')['Revenue_Impact_USD'].sum().reset_index()

# 2. Sort the values so the chart goes from highest to lowest (Best Practice!)
revenue_by_dept = revenue_by_dept.sort_values(by='Revenue_Impact_USD', ascending=False)

# 3. Create the Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(
    data=revenue_by_dept,
    x='Department',
    y='Revenue_Impact_USD',
    palette='viridis' # A colorblind-friendly, professional color palette
)

# 4. Add clear titles and labels for the non-technical users
plt.title('Total Revenue Impact by Department', fontsize=16, fontweight='bold')
plt.xlabel('Department', fontsize=12)
plt.ylabel('Total Revenue Impact (USD)', fontsize=12)
plt.xticks(rotation=45) # Rotates the text so department names don't overlap

# 5. Render the chart
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional theme for all plots
sns.set_theme(style="whitegrid")

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# -----------------------------------------------------------------
# 1. The Business Overview: Revenue Impact by Department (Top Left)
# -----------------------------------------------------------------
revenue_by_dept = df.groupby('Department')['Revenue_Impact_USD'].sum().reset_index()
revenue_by_dept = revenue_by_dept.sort_values(by='Revenue_Impact_USD', ascending=False)

sns.barplot(
    data=revenue_by_dept, x='Department', y='Revenue_Impact_USD',
    palette='viridis', ax=axes[0, 0]
)
axes[0, 0].set_title('Total Revenue Impact by Department', fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Department', fontsize=12)
axes[0, 0].set_ylabel('Total Revenue Impact (USD)', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# -----------------------------------------------------------------
# 2. The Risk Dashboard: Anomalies by Priority Level (Top Right)
# -----------------------------------------------------------------
sns.countplot(
    data=df, x='Priority_Level', hue='Anomaly_Detected',
    palette='Set2', order=['Low', 'Medium', 'High', 'Critical'],
    ax=axes[0, 1]
)
axes[0, 1].set_title('Anomaly Detection Breakdown by Priority Level', fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Priority Level', fontsize=12)
axes[0, 1].set_ylabel('Number of Transactions', fontsize=12)
axes[0, 1].legend(title='Anomaly Detected')

# -----------------------------------------------------------------
# 3. The Operations Tracker: Query Volume Over Time (Bottom Left)
# -----------------------------------------------------------------
# Count the number of transactions per day
daily_volume = df.groupby(df['Transaction_Date'].dt.date).size().reset_index(name='Query_Count')
daily_volume = daily_volume.sort_values(by='Transaction_Date')

sns.lineplot(
    data=daily_volume, x='Transaction_Date', y='Query_Count',
    marker='o', color='b', linewidth=2, ax=axes[1, 0]
)
axes[1, 0].set_title('Daily AI Agent Query Volume', fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Date', fontsize=12)
axes[1, 0].set_ylabel('Total Queries Processed', fontsize=12)
axes[1, 0].tick_params(axis='x', rotation=45)

# -----------------------------------------------------------------
# 4. The System Health Check: Complexity vs. Processing Time (Bottom Right)
# -----------------------------------------------------------------
sns.scatterplot(
    data=df, x='Query_Complexity_Score', y='Processing_Time_Sec',
    hue='SLA_Breached', palette={False: 'green', True: 'red'}, alpha=0.7,
    ax=axes[1, 1]
)
axes[1, 1].set_title('Agent Efficiency: Query Complexity vs. Processing Time', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Query Complexity Score (0.0 to 1.0)', fontsize=12)
axes[1, 1].set_ylabel('Processing Time (Seconds)', fontsize=12)
axes[1, 1].legend(title='SLA Breached')

# -----------------------------------------------------------------
# Render the Dashboard
# -----------------------------------------------------------------
plt.tight_layout()
plt.show()
