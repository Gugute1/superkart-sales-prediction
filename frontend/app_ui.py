import os
import pandas as pd
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon=":material/shopping_cart:",           # shopping cart icon from Google's Material Symbols library
    layout="wide"
)

st.title("SuperKart Store Sales Prediction Dashboard")
st.markdown("Predict product-level store revenue using a hypertuned CatBoost model via the FastAPI backend.")

# Sidebar: Backend API Configuration
st.sidebar.header("Backend Connection")
default_url = os.getenv("BACKEND_URL", "http://localhost:8000")
backend_url = st.sidebar.text_input("FastAPI Base URL", value=default_url).rstrip("/")
st.sidebar.caption("Update this URL to the deployed FastAPI endpoint when live.")

# Three tabs matching the FastAPI single and batch endpoints
tab_single, tab_json_list, tab_batch = st.tabs(["Single Item Prediction", "Batch JSON List Prediction", "Batch Prediction (CSV Upload)"])

# =====================================================================
# TAB 1: SINGLE PRODUCT PREDICTION (/v1/predict)
# =====================================================================
with tab_single:
    st.subheader("Enter Product & Store Details")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        # ----------------------------------
        #    Product_Weight: float
        #    Product_Sugar_Content: str
        #    Product_Allocated_Area: float
        # ----------------------------------
        with col1:
            product_weight = st.number_input(
                "Product Weight",
                min_value=4.0,
                max_value=22.0,
                value=12.66,
                step=0.1
            )
            product_sugar = st.selectbox(
                "Product Sugar Content",
                options=["Low Sugar", "Regular", "No Sugar"]
            )
            product_area = st.slider(
                "Product Allocated Area",
                min_value=0.004,
                max_value=0.298,
                value=0.056,
                step=0.005,
                format="%.3f"
            )

        # ----------------------------------
        #    Product_MRP: float
        #    Store_Size: str
        #    Store_Location_City_Type: str
        # ----------------------------------
        with col2:
            product_mrp = st.number_input(
                "Product MRP ($)",
                min_value=31.0,
                max_value=266.0,
                value=146.74,
                step=1.0
            )
            store_size = st.selectbox(
                "Store Size",
                options=["Small", "Medium", "High"]
            )
            store_city_type = st.selectbox(
                "Store Location City Type",
                options=["Tier 1", "Tier 2", "Tier 3"]
            )

        # ----------------------------------
        #    Store_Type: str
        #    Product_ID_Char: str
        #    Store_Age_Years: int
        #    Product_Type_Category: str
        # ----------------------------------
        with col3:
            store_type = st.selectbox(
                "Store Type",
                options=["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]               
            )
            product_id_char = st.selectbox(
                "Product ID Prefix (Product_ID_Char)",
                options=["FD", "DR", "NC"]
            )
            store_age = st.slider(
                "Store Age (Years)",
                min_value=17,
                max_value=39,
                value=17
            )
            product_category = st.selectbox(
                "Product Type Category",
                options=["Perishables", "Non Perishables"]
            )

        submitted = st.form_submit_button("Predict Store Sales", use_container_width=True)

    if submitted:

        payload = {
            "Product_Weight": float(product_weight),
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": float(product_area),
            "Product_MRP": float(product_mrp),
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city_type,
            "Store_Type": store_type,
            "Product_ID_Char": product_id_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": product_category
        }

        try:

            with st.spinner("Querying FastAPI backend..."):
                response = requests.post(f"{backend_url}/v1/predict", json=payload, timeout=15)

            if response.status_code == 200:
                result = response.json()
                pred_val = result["predicted_sales_total"]
                st.success("Prediction retrieved successfully!")
                st.metric(label="Predicted Total Store Sales (USD)", value=f"${pred_val:,.2f}")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Connection failed. Verify your FastAPI server is running at {backend_url}.\n\nDetails: {e}")


# =====================================================================
# TAB 2: Batch JSON List Prediction (/v1/predict_json_list)
# =====================================================================
with tab_json_list:
    st.subheader("Batch Prediction via JSON List")
    st.markdown(
        "Paste or edit a JSON list of product objects containing keys: `Product_Weight`, "
        "`Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`, `Store_Size`, "
        "`Store_Location_City_Type`, `Store_Type`, `Product_ID_Char`, `Store_Age_Years`, "
        "and `Product_Type_Category`."
    )

    sample_json_list = """[
    {
        "Product_Weight": 12.66,
        "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.056,
        "Product_MRP": 146.74,
        "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type2",
        "Product_ID_Char": "FD",
        "Store_Age_Years": 17,
        "Product_Type_Category": "Perishables"
    },
    {
        "Product_Weight": 8.20,
        "Product_Sugar_Content": "Regular",
        "Product_Allocated_Area": 0.042,
        "Product_MRP": 89.50,
        "Store_Size": "High",
        "Store_Location_City_Type": "Tier 2",
        "Store_Type": "Supermarket Type1",
        "Product_ID_Char": "DR",
        "Store_Age_Years": 39,
        "Product_Type_Category": "Non Perishables"
    }
]"""

    json_input_str = st.text_area(
        "JSON Array Payload",
        value=sample_json_list,
        height=300
    )

    if st.button("Run JSON List Predictions", use_container_width=True):
        try:
            records_payload = json.loads(json_input_str)

            if not isinstance(records_payload, list) or len(records_payload) == 0:
                st.error("Input must be a non-empty JSON list of product objects (`[...]`).")
            else:
                required_cols = [
                    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
                    "Product_MRP", "Store_Size", "Store_Location_City_Type",
                    "Store_Type", "Product_ID_Char", "Store_Age_Years", "Product_Type_Category"
                ]

                batch_df = pd.DataFrame(records_payload)
                missing_cols = [c for c in required_cols if c not in batch_df.columns]

                if missing_cols:
                    st.error(f"Missing required keys in JSON objects: {missing_cols}")
                else:
                    with st.spinner(f"Sending {len(records_payload)} JSON records to FastAPI..."):
                        response = requests.post(
                            f"{backend_url}/v1/predict_json_list",
                            json=records_payload,
                            timeout=60
                        )

                    if response.status_code == 200:
                        res_json = response.json()
                        preds = [item["predicted_sales_total"] for item in res_json["predictions"]]

                        results_df = batch_df.copy()
                        results_df["Predicted_Sales_Total_USD"] = preds

                        st.success(f"Completed predictions for {res_json['count']} items!")
                        st.dataframe(results_df, use_container_width=True)

                        json_bytes = results_df.to_json(orient="records", indent=2).encode("utf-8")
                        st.download_button(
                            label="Download Predictions JSON",
                            data=json_bytes,
                            file_name="superkart_json_batch_predictions.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

        except json.JSONDecodeError as je:
            st.error(f"Invalid JSON syntax: {je}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection failed. Verify your FastAPI server is running at {backend_url}.\n\nDetails: {e}")


# =====================================================================
# TAB 3: Batch File Prediction (/v1/predict_batch)
# =====================================================================
with tab_batch:
    st.subheader("Upload Batch CSV File")
    st.markdown(
        "Upload a `.csv` file containing columns: `Product_Weight`, `Product_Sugar_Content`, "
        "`Product_Allocated_Area`, `Product_MRP`, `Store_Size`, `Store_Location_City_Type`, "
        "`Store_Type`, `Product_ID_Char`, `Store_Age_Years`, and `Product_Type_Category`."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_df)}** rows. Preview:")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("Run Batch Predictions", use_container_width=True):
            required_cols = [
                "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
                "Product_MRP", "Store_Size", "Store_Location_City_Type",
                "Store_Type", "Product_ID_Char", "Store_Age_Years", "Product_Type_Category"
            ]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                # Package the validated DataFrame into an in-memory CSV file upload tuple
                csv_payload = batch_df[required_cols].to_csv(index=False).encode("utf-8")
                files_payload = {
                    "file": (uploaded_file.name, csv_payload, "text/csv")
                }

                try:
                    with st.spinner(f"Uploading {len(batch_df)} rows to FastAPI..."):
                        response = requests.post(
                            f"{backend_url}/v1/predict_batch",
                            files=files_payload,
                            timeout=60
                        )

                    if response.status_code == 200:
                        res_json = response.json()
                        preds = [item["predicted_sales_total"] for item in res_json["predictions"]]

                        results_df = batch_df.copy()
                        results_df["Predicted_Sales_Total_USD"] = preds

                        st.success(f"Completed predictions for {res_json['count']} items!")
                        st.dataframe(results_df, use_container_width=True)

                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_bytes,
                            file_name="superkart_batch_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection failed. Verify your FastAPI server is running at {backend_url}.\n\nDetails: {e}")
