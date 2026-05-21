import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

def show_nlp_dashboard():

    # LOAD DATA
    df_epo = pd.read_csv('Dataset/NLP/df_epo_raw.csv')
    df_llm = pd.read_csv('Dataset/NLP/df_llm_raw.csv')

    df = pd.concat([df_epo, df_llm], ignore_index=True)

    #LOAD ASSETS
    emotion_colors = {
        "happy": "#00CC96",
        "sad": "#636EFA",
        "angry": "#EF553B",
        "anxious": "#AB63FA",
        "neutral": "#FFA15A"
    }

    emotion_images = {
        "happy": "assets/happy.png",
        "sad": "assets/sad.png",
        "angry": "assets/angry.png",
        "anxious": "assets/anxious.png",
        "neutral": "assets/neutral.png"
    }

    # SIDEBAR FILTER
    st.sidebar.subheader("NLP Filter")

    selected_emotion = st.sidebar.multiselect(
        "Filter Emotion",
        df['emotion_label'].unique(),
        default=df['emotion_label'].unique()
    )

    df = df[df['emotion_label'].isin(selected_emotion)]

    # TITLE
    total_data = len(df)
    dominant_emotion = df['emotion_label'].mode()[0]
    avg_stress = round(df['stress_label'].mean(), 2)

    col1, col2 = st.columns([5, 1])

    with col1:
        st.title("🗣️ NLP Emotion Dashboard")
        st.caption("Emotion & Stress Analysis NLP Dataset")
    
    with col2:
        st.image(emotion_images[dominant_emotion], width=200)
    
    
    # METRICS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Total Data", f"{total_data:,}")

    with col2:
        st.metric("😊 Dominant Emotion", dominant_emotion)

    with col3:
        st.metric("🔥 Average Stress", avg_stress)

    st.divider()

    # EMOTION DISTRIBUTION
    emotion_count = df['emotion_label'].value_counts().reset_index()
    emotion_count.columns = ['Emotion', 'Count']

    # BAR CHART
    fig_bar = px.bar(
        emotion_count,
        x='Emotion',
        y='Count',
        color='Emotion',
        color_discrete_map=emotion_colors,
        title='Emotion Distribution'
    )

    fig_bar.update_layout(template='plotly_dark')

    # PIE CHART
    fig_pie = px.pie(
        emotion_count,
        names='Emotion',
        values='Count',
        color='Emotion',
        color_discrete_map=emotion_colors,
        hole=0.4,
        title='Emotion Percentage'
    )

    fig_pie.update_layout(template='plotly_dark')

    # DISPLAY CHARTS
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # STRESS DISTRIBUTION
    fig_hist = px.histogram(
        df,
        x='stress_label',
        nbins=20,
        title='Stress Distribution'
    )

    fig_hist.update_layout(template='plotly_dark')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()

    # WORD CLOUD
    st.subheader("☁️ Word Cloud")
    col1, col2, col3 = st.columns([1,3,1])

    with col1:
        st.image("assets/anxious.png", width=90)
    
    with col2:
        text = " ".join(df['text'].astype(str)).lower()

        custom_stopwords = {
            'aku', 'saya', 'dan', 'yang', 'orang', 'rumah', 'di rumah',
            'di', 'ke', 'dari', 'itu', 'tadi', 'pagi', 
            'ini', 'aja', 'banget', 'sama', 'keluarga', 'hari', 'waktu', 'kampus', 'malam'
            'karena', 'untuk', 'temen', 'kantor', 'dia', 'perjalanan', 
        }

        wordcloud = WordCloud(
            width=1200,
            height=500,
            background_color='white',
            stopwords=custom_stopwords
        ).generate(text)

        fig_wc, ax = plt.subplots(figsize=(15, 5))

        ax.imshow(wordcloud, interpolation='bilinear')

        ax.axis('off')
        st.pyplot(fig_wc)

    with col3:
        st.image("assets/happy.png", width=90)

    st.divider()

    # SAMPLE TEXT BY EMOTION
    st.subheader("📝 Sample Text by Emotion")

    emotions = df['emotion_label'].unique()
    cols = st.columns(len(emotions))

    for i, emotion in enumerate(emotions):
        sample_text = df[df['emotion_label'] == emotion]['text'].sample(3).tolist()

        with cols[i]:
            st.image(emotion_images[emotion], width=80)
            st.markdown(f"### {emotion}")
            for text in sample_text:
                with st.container(border=True):
                    st.caption(f"- {text[:80]}...")

    st.divider()

    st.caption("Vitara Capstone Project • NLP Emotion Analysis Dashboard")