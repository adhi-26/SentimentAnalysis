import streamlit as st
import pandas
import plotly.graph_objects as go
from ast import literal_eval
import matplotlib.pyplot as plt

def PiePlot(labels, values):
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = [value for value in values],
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
    st.plotly_chart(fig, use_container_width=True)

def RenderPage(movie, predictions, wordcloud):
    col1, col2 = st.columns([1,2])
    with col1:
        st.image(movie['poster'], width=120)
    with col2:
        st.markdown(f"**{movie['title']}** (**{int(movie['release-year'])}**)")
        st.markdown(f"{movie['genre']}")
        director = ', '.join([item['name'] for item in literal_eval(movie['directors'])])
        st.markdown(f"Director: {director}")
        cast = ', '.join([item['name'] for item in literal_eval(movie['cast'])])
        st.markdown(f"Cast: {cast}")
    
    st.text('')
    st.markdown('Plot:')
    st.markdown(movie['overview'])

    col3, col4 = st.columns([1, 2])
    positive= round(100*list(predictions).count("pos")/len(predictions))
    with col3:
        st.subheader('Analysis:')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown(f'<div style="text-align:center;"><span style="font-size:120px;padding-top:5px">{positive}</span>%</div', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:20px; text-align:center;">Positive Reviews</div>', unsafe_allow_html=True)
    with col4:
        PiePlot(['Positive', 'Negative'], [list(predictions).count('pos'), list(predictions).count('neg')])

    fig, ax = plt.subplots(figsize=(15,15))
    fig.set_facecolor('k')
    ax.imshow(wordcloud)
    ax.axis('off')
    fig.set_in_layout(False)
    st.pyplot(fig)

    


