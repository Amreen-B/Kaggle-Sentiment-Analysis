import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sentiment Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🤖 Sentiment Intelligence Platform")

st.caption(
    "Transformer-powered sentiment analysis using FastAPI + DistilBERT"
)

# =========================================================
# TOP SECTION
# =========================================================

top_left, top_right = st.columns([1, 1])

# =========================================================
# SINGLE TEXT ANALYSIS
# =========================================================

with top_left:

    st.subheader("🔍 Single Text Analysis")

    user_input = st.text_area(
        "Enter text",
        height=10
    )

    if st.button("Analyze Sentiment"):

        if user_input.strip():

            with st.spinner("Analyzing..."):

                try:

                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json={"text": user_input}
                    )

                    result = response.json()

                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    # Neutral threshold
                    if confidence < 0.60:
                        sentiment = "NEUTRAL"

                    # Sentiment box
                    if sentiment == "POSITIVE":

                        st.markdown("""
                        <div style="
                            background:#052e16;
                            border:1px solid #22c55e;
                            padding:18px;
                            border-radius:12px;
                            text-align:center;
                            font-size:24px;
                            font-weight:bold;
                            color:#22c55e;
                        ">
                        😊 POSITIVE
                        </div>
                        """, unsafe_allow_html=True)

                    elif sentiment == "NEGATIVE":

                        st.markdown("""
                        <div style="
                            background:#450a0a;
                            border:1px solid #ef4444;
                            padding:18px;
                            border-radius:12px;
                            text-align:center;
                            font-size:24px;
                            font-weight:bold;
                            color:#ef4444;
                        ">
                        😠 NEGATIVE
                        </div>
                        """, unsafe_allow_html=True)

                    else:

                        st.markdown("""
                        <div style="
                            background:#172554;
                            border:1px solid #3b82f6;
                            padding:18px;
                            border-radius:12px;
                            text-align:center;
                            font-size:24px;
                            font-weight:bold;
                            color:#3b82f6;
                        ">
                        😐 NEUTRAL
                        </div>
                        """, unsafe_allow_html=True)

                    st.progress(float(confidence))

                    st.write(f"Confidence Score: {confidence}")

                except Exception as e:
                    st.error(f"Error: {e}")

# =========================================================
# BATCH CSV ANALYSIS
# =========================================================

with top_right:

    st.subheader("📂 Batch CSV Analysis")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

# =========================================================
# PROCESS CSV
# =========================================================

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # Select text column
        text_column = st.selectbox(
            "Select text column",
            df.columns
        )

        sentiments = []
        confidences = []

        with st.spinner("Processing predictions..."):

            for text in df[text_column]:

                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": str(text)}
                )

                result = response.json()

                sentiment = result["sentiment"]
                confidence = result["confidence"]

                if confidence < 0.60:
                    sentiment = "NEUTRAL"

                sentiments.append(sentiment)
                confidences.append(confidence)

        # Add prediction columns
        df["Sentiment"] = sentiments
        df["Confidence"] = confidences

        # =========================================================
        # MAIN DASHBOARD
        # =========================================================

        left_col, right_col = st.columns([2.2, 1])

        # =========================================================
        # LEFT SIDE — RESULTS TABLE
        # =========================================================

        with left_col:

            st.subheader("Prediction Results")

            st.dataframe(
                df,
                use_container_width=True,
                height=520
            )

        # =========================================================
        # RIGHT SIDE — ANALYTICS
        # =========================================================

        with right_col:

            sentiment_counts = df["Sentiment"].value_counts()

            st.subheader("Sentiment Distribution")

            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                hole=0.45,
                color=sentiment_counts.index,
                color_discrete_map={
                    "POSITIVE": "#22c55e",
                    "NEGATIVE": "#ef4444",
                    "NEUTRAL": "#3b82f6"
                }
            )

            fig.update_layout(
                height=350,
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =====================================================
            # SUMMARY METRICS
            # =====================================================

            positive_count = sentiment_counts.get("POSITIVE", 0)
            negative_count = sentiment_counts.get("NEGATIVE", 0)
            neutral_count = sentiment_counts.get("NEUTRAL", 0)

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("😊 Positive", positive_count)

            with m2:
                st.metric("😠 Negative", negative_count)

            with m3:
                st.metric("😐 Neutral", neutral_count)

            with m4:
                st.metric("📄 Total", len(df))

    except Exception as e:
        st.error(f"Error processing file: {e}")