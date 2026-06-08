import streamlit as st
import joblib 
import pandas as pd
model=joblib.load("house_price_model.joblib")
st.set_page_config(page_title="House Price Preidction")
st.title("house Price Prediction")
st.write("Upload a csv file and predict house price")
uploaded_file=st.file_uploader("Upload CSV",type=["csv"])
if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.subheader("Input Date")
    st.dataframe(data.head())
    if st.button("Predict"):
        predictions=model.predict(data)
        result=data.copy()
        result["PredictedPrice"]=predictions
        st.subheader("Predictions")
        st.dataframe(result)
        csv=result.to_csv(index=False)
        st.download_button("Download Predictions",csv,"predictions.csv","text/cvs")
