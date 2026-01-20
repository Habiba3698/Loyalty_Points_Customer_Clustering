import streamlit as st
import numpy as np
import pandas as pd
import joblib  


st.title("Customer Cluster and Top Merchants Recommender")

# load our clustered dataset with cluster names
df = pd.read_parquet("Data/clustered/final_clustered_customer_df.parquet", engine='pyarrow')  

# load our model to predict if adding new customers
model = joblib.load("Models/kmeans_customer_model.pkl")  

# add our mapping in case we have a new customer
cluster_names = {
    0: "Occasional Low-Value",   # High recency, low spend
    1: "New / Recent",           # Low recency, single transaction
    2: "High-Value Occasional",  # High spend, lower frequency
    3: "Premium / VIP",          # Extreme high spend & frequency
    4: "Dormant / Low-Value",    # High recency, low spend
    5: "Regular Engaged",        # High frequency, solid spend
    6: "Active Low-Value"        # Low recency, multiple low-spend visits
}

st.write("Enter a Customer ID to see their cluster and top merchants, or input a new customer's features to get merchant recommendations.")
type = st.radio("New or Existing Customer? :", ["Existing Customer", "New Customer"])

if type == "Existing Customer":    # if customer already exists
    id = st.selectbox("Select Customer ID", df['User_Id'].unique())
    customer = df[df['User_Id'] == id].iloc[0]

    st.subheader(f"Profile: {customer['Cluster_Name']}")
    st.write(f"Total Spend: ${customer['Total_Spend']:,.2f}")
    st.write(f"Avg Transaction: ${customer['Avg_Trx_Value']:,.2f}")
    st.write(f"Transactions: {customer['Trx_Count']}")
    st.write(f"Recency: {customer['Recency']} days")
    st.write(f"Points Balance: {customer['Total_Points']:,}")

    # Top merchants
    merchants = [customer[f"Top_Merchant_{i}"] for i in range(1,5) if customer[f"Top_Merchant_{i}"] != "Unknown"]
    st.subheader("🏆 Top Merchants")

    if merchants:
        st.markdown(" • " + "\n • ".join(merchants))
    else:
        st.write("No merchant recommendations for this cluster.")


else:   # new customer input 
    st.subheader("Enter New Customer Features")

    # numerical features
    trx_count = st.number_input("Transaction Count", min_value=0, value=1)
    total_spend = st.number_input("Total Spend", min_value=0.0, value=100.0)
    avg_trx_value = st.number_input("Average Transaction Value", min_value=0.0, value=100.0)
    recency = st.number_input("Recency (days since last transaction)", min_value=0.0, value=30.0)
    total_points = st.number_input("Total Points", min_value=0.0, value=1000.0)

    if st.button("Predict Cluster & Recommend Merchants"):
        # handle categorical features since model expects 9 features
        new_customer = pd.DataFrame([[trx_count, total_spend, avg_trx_value, recency, total_points,
                                      "Unknown", "Unknown", "Unknown", "Unknown"]],     
                                    columns=['Trx_Count','Total_Spend','Avg_Trx_Value','Recency','Total_Points',
                                             'Top_Merchant_1','Top_Merchant_2','Top_Merchant_3','Top_Merchant_4'])

        # Predict new customer cluster
        cluster = model.predict(new_customer)[0]
        cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
        st.success(f"Predicted Cluster: {cluster_name}")

        # Recommend top merchants from this cluster
        cluster_customers = df[df['Cluster'] == cluster]
        merchant_columns = ["Top_Merchant_1","Top_Merchant_2","Top_Merchant_3","Top_Merchant_4"]
        merchants = cluster_customers[merchant_columns].values.flatten()
        recommended_merchants = pd.Series(merchants)
        recommended_merchants = recommended_merchants[recommended_merchants != "Unknown"]
        top_merchants = recommended_merchants.value_counts().index.tolist()

        st.subheader("Recommended Top Merchants")
        if top_merchants:
            st.markdown(" • " + "\n • ".join(top_merchants))
        else:
            st.write("No merchant recommendations for this cluster.")