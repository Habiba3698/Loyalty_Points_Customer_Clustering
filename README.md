# Customer Cluster and Top Merchants Recommender

A machine learning-powered Streamlit application that clusters customers and provides personalized merchant recommendations based on their spending behavior and transaction patterns.

## 📋 Overview

This project uses K-Means clustering to segment customers into 7 distinct behavioral groups and recommends top merchants for each cluster. It supports both existing customer lookup and new customer prediction.

## 🎯 Customer Segments

The model identifies 7 customer clusters:

| Cluster | Name | Profile |
|---------|------|---------|
| 0 | Occasional Low-Value | High recency, low spend |
| 1 | New / Recent | Low recency, single transaction |
| 2 | High-Value Occasional | High spend, lower frequency |
| 3 | Premium / VIP | Extreme high spend & frequency |
| 4 | Dormant / Low-Value | High recency, low spend |
| 5 | Regular Engaged | High frequency, solid spend |
| 6 | Active Low-Value | Low recency, multiple low-spend visits |

## 📁 Project Structure

```
points_app.py                                    # Main Streamlit application
Data/
├── clustered/
│   └── final_clustered_customer_df.parquet     # Clustered customer dataset
├── merged/
└── raw/
Models/
└── kmeans_customer_model.pkl                   # Trained K-Means clustering model
README.md                                        # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Joblib

### Installation

1. Navigate to your project directory
2. Install required dependencies:

```bash
pip install streamlit pandas numpy joblib
```

### Running the Application

```bash
streamlit run points_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 💡 Features

### For Existing Customers
- Search and select customer by ID from dropdown
- View detailed customer profile:
  - Assigned cluster classification
  - Total spend and average transaction value
  - Transaction count
  - Recency (days since last transaction)
  - Points balance
- Get top 4 merchant recommendations for their cluster

### For New Customers
- Input custom features:
  - Transaction count
  - Total spend
  - Average transaction value
  - Recency (days since last transaction)
  - Total points
- Get cluster prediction
- Receive merchant recommendations based on similar customers in that cluster

## 📊 Key Metrics

The clustering model uses the following customer features:

| Feature | Description |
|---------|-------------|
| `Trx_Count` | Total number of transactions |
| `Total_Spend` | Total spending amount |
| `Avg_Trx_Value` | Average transaction value |
| `Recency` | Days since last transaction |
| `Total_Points` | Total loyalty points balance |

## 🔄 How It Works

### Existing Customer Flow
1. User selects a customer ID from the dropdown
2. App loads customer data from clustered dataset
3. Displays customer profile (cluster, spend metrics, recency, points)
4. Shows top 4 merchants for that customer's cluster

### New Customer Prediction Flow
1. User inputs 5 numerical features
2. Features are formatted into DataFrame with 9 columns (5 numeric + 4 merchant placeholders)
3. Trained K-Means model predicts the cluster
4. App aggregates top merchants from all customers in that cluster
5. Recommends merchants ranked by frequency within the cluster

## 🛠️ Technical Details

- **Algorithm**: K-Means Clustering
- **Number of Clusters**: 7
- **Model File**: `Models/kmeans_customer_model.pkl`
- **Data Format**: Parquet (optimized for large datasets)
- **Framework**: Streamlit (fast, interactive web apps)

## 📝 Notes

- Merchant recommendations marked as "Unknown" are automatically filtered out
- Top merchants are ranked by frequency across all customers in the same cluster
- New customer predictions are based on K-Means cluster assignment
- The model expects 9 columns for predictions (5 numeric + 4 merchant columns)

## 👤 Author

Habiba Derbala

## 📄 License

[Add your license information here]