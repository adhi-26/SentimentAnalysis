import streamlit as st
import pandas as pd

def RenderPage(reviews: pd.DataFrame):
    st.subheader('Top Reviews')
    num = min(10, len(reviews))
    for i in range(num):
        st.markdown(f"**{reviews.iloc[i]['Title']}**")
        st.markdown(reviews.iloc[i]['Review'])
        st.text("")