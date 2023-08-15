import streamlit as st
from streamlit_option_menu import option_menu

def show():
    with st.sidebar:
        st.markdown('Movie Reviews', unsafe_allow_html=False)
        selected = option_menu(
                menu_title=None,
                options=['Movie Reviews'],
                icons=['Film'],
                default_index=0
        )
        return selected
                
        
        