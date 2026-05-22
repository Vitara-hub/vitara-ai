import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def show_keystroke_dashboard():

    # LOAD DATA
    df = pd.read_csv('https://drive.google.com/uc?id=1BMdcEBx4x2-KE4HhOjdvAjIQny3BXDVP')

    # SIDEBAR FILTER
    st.sidebar.subheader("⌨️ Keystroke Filter")

    stress_range = st.sidebar.slider(
        "Stress Range",
        0.0,
        1.0,
        (0.0, 1.0)
    )

    df = df[
        (df['stress_label'] >= stress_range[0]) &
        (df['stress_label'] <= stress_range[1])
    ]

    # HEADER
    col1, col2 = st.columns([5,1])

    with col1:
        st.title("⌨️ Keystroke Dynamics Dashboard")
        st.caption("Typing Behavior & Stress Pattern Analysis")

    # with col2:
    #     st.image("Assets/keyboard.png", width=120)

    # METRICS
    total_users = df['user_id'].nunique()
    total_sessions = len(df)

    avg_wpm = round(df['wpm'].mean(), 1)

    avg_stress = round(df['stress_label'].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👤 Total Users", total_users)

    with col2:
        st.metric("📄 Total Sessions", f"{total_sessions:,}")

    with col3:
        st.metric("⚡ Average WPM", avg_wpm)

    with col4:
        st.metric("🔥 Average Stress", avg_stress)

    st.divider()

    # WPM DISTRIBUTION
    fig_wpm = px.histogram(
        df,
        x='wpm',
        nbins=20,
        title='Typing Speed Distribution',
        color_discrete_sequence=['#00F5FF']
    )

    fig_wpm.update_layout(
        template='plotly_dark',
        height=400
    )

    # STRESS DISTRIBUTION
    fig_stress = px.histogram(
        df,
        x='stress_label',
        nbins=20,
        title='Stress Score Distribution',
        color_discrete_sequence=['#FF4ECD']
    )

    fig_stress.update_layout(
        template='plotly_dark',
        height=400
    )

    # ROW 1
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_wpm, use_container_width=True)

    with col2:
        st.plotly_chart(fig_stress, use_container_width=True)

    # TOP FASTEST USERS
    top_users = (
        df.groupby('user_id')['wpm']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_fastest = px.bar(
        top_users,
        x='user_id',
        y='wpm',
        color='wpm',
        color_continuous_scale='Turbo',
        title='Top Fastest Users'
    )

    fig_fastest.update_layout(
        template='plotly_dark',
        height=400
    )

    # WPM VS STRESS
    fig_scatter = px.scatter(
        df,
        x='wpm',
        y='stress_label',
        color='typing_variance',
        size='backspace_rate',
        hover_data=['user_id'],
        color_continuous_scale='Viridis',
        title='Typing Speed vs Stress'
    )

    fig_scatter.update_layout(
        template='plotly_dark',
        height=400
    )

    # ROW 2
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_fastest, use_container_width=True)

    with col2:
        st.plotly_chart(fig_scatter, use_container_width=True)

    # TYPING VARIANCE
    fig_variance = px.box(
        df,
        y='typing_variance',
        title='Typing Variance Distribution',
        color_discrete_sequence=['#8B5CF6']
    )

    fig_variance.update_layout(
        template='plotly_dark',
        height=400
    )

    st.plotly_chart(fig_variance, use_container_width=True)

    # INSIGHT BOX
    if avg_wpm > 60:
        st.success("⚡ Users show fast typing behavior with stable performance.")

    elif avg_wpm > 40:
        st.info("⌨️ Users show moderate typing performance.")

    else:
        st.warning("🐢 Users show slow typing behavior.")

    # DECORATION
    # col1, col2, col3 = st.columns([1,3,1])

    # with col1:
    #     st.image("", width=90)

    # with col3:
    #     st.image("", width=90)

    # USER SUMMARY
    st.subheader("👤 User Typing Summary")

    user_summary = (
        df.groupby('user_id')
        .agg({
            'wpm':'mean',
            'stress_label':'mean',
            'backspace_rate':'mean'
        })
        .reset_index()
    )

    st.dataframe(
        user_summary,
        use_container_width=True
    )

    # FOOTER
    st.divider()

    st.caption("Vitara Capstone Project • Keystroke Dynamics Dashboard")