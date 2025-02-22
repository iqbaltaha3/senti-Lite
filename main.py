import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import time
import io
import pandas as pd

# Download necessary NLTK resources quietly
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Added to ensure punkt_tab is available

# Instantiate the SentimentIntensityAnalyzer globally so it isn’t cached
sia = SentimentIntensityAnalyzer()

@st.cache_data
def analyze_sentiment(text):
    # Tokenize sentences using NLTK's Punkt tokenizer
    sentences = nltk.sent_tokenize(text)
    pos_sentences = []
    neg_sentences = []
    neutral_sentences = []
    overall_score = 0
    pos_total, neg_total, neu_total = 0, 0, 0
    sentence_data = []
    
    for sentence in sentences:
        sentiment_score = sia.polarity_scores(sentence)
        overall_score += sentiment_score['compound']
        pos_total += sentiment_score['pos']
        neg_total += sentiment_score['neg']
        neu_total += sentiment_score['neu']
        compound = sentiment_score['compound']
        classification = "Neutral"
        if compound >= 0.05:
            pos_sentences.append(sentence)
            classification = "Positive"
        elif compound <= -0.05:
            neg_sentences.append(sentence)
            classification = "Negative"
        else:
            neutral_sentences.append(sentence)
        
        sentence_data.append({
            "Sentence": sentence, 
            "Compound Score": compound, 
            "Classification": classification
        })
    
    total_sentences = len(sentences) if sentences else 1
    return {
        "overall": overall_score / total_sentences,
        "pos_percent": (pos_total / total_sentences) * 100,
        "neg_percent": (neg_total / total_sentences) * 100,
        "neu_percent": (neu_total / total_sentences) * 100,
        "pos_sentences": pos_sentences,
        "neg_sentences": neg_sentences,
        "sentence_data": sentence_data
    }

def highlight_text(sentence_data):
    highlighted = ""
    for data in sentence_data:
        sentence = data["Sentence"]
        classification = data["Classification"]
        if classification == "Positive":
            color = "#d4edda"  # light green
        elif classification == "Negative":
            color = "#f8d7da"  # light red
        else:
            color = "#d1ecf1"  # light blue for neutral
        highlighted += f'<span style="background-color: {color}; padding: 3px; margin:2px; border-radius: 3px;">{sentence}</span> '
    return highlighted

st.sidebar.title("senti-Lite")
app_mode = st.sidebar.selectbox("Navigation", ["Sentiment Analysis", "About"])

if app_mode == "About":
    st.title("senti-Lite")
    st.markdown("""
        **SparrowSentiment** is an AI-powered sentiment analysis tool that lets you dive deep into the emotional tone of your text.
        
        **Key Features:**
        - **Flexible Input:** Analyze a single text or compare two texts side by side.
        - **Detailed Analysis:** Get overall sentiment, sentence-level scores, and a breakdown by sentiment type.
        - **Visual Insights:** View your data with pie charts, bar charts, and histograms.
        - **Inline Highlights:** See your text color-coded by sentiment.
        - **Downloadable Reports:** Export your analysis as TXT and CSV files.
    """)
    st.stop()

st.title("🕊 senti-Lite: Comprehensive Sentiment Analysis")
st.markdown("Analyze the sentiment of your text with AI-powered insights!")

analysis_mode = st.sidebar.radio("Analysis Mode", ["Single Analysis", "Comparison Analysis"])

#############################
# Single Analysis Mode
#############################
if analysis_mode == "Single Analysis":
    st.subheader("Single Text Analysis")
    input_method = st.sidebar.radio("Input Method", ("Text Input", "File Upload"))
    sample_text = (
        "I love this product! It has changed my life. However, the shipping was delayed and the packaging was terrible. "
        "Overall, I'm happy with the purchase, but there is room for improvement."
    )
    
    if input_method == "Text Input":
        text_input = st.text_area("Enter text for sentiment analysis:", height=150, value=sample_text)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a text file", type=["txt"])
        text_input = str(uploaded_file.read(), 'utf-8') if uploaded_file else ""
    
    if st.button("🔍 Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing..."):
                time.sleep(1)
                result = analyze_sentiment(text_input)
                overall_score = result["overall"]
                pos_percent = result["pos_percent"]
                neg_percent = result["neg_percent"]
                neu_percent = result["neu_percent"]
                sentence_data = result["sentence_data"]
                pos_sentences = result["pos_sentences"]
                neg_sentences = result["neg_sentences"]
                
                total_sentences = len(sentence_data)
                count_positive = len(pos_sentences)
                count_negative = len(neg_sentences)
                count_neutral = total_sentences - count_positive - count_negative
                
                sentiment = "Positive" if overall_score >= 0.05 else "Negative" if overall_score <= -0.05 else "Neutral"
                st.success(f"Overall Sentiment: **{sentiment}** (Score: {overall_score:.2f})")
            
            st.subheader("Summary")
            st.markdown(f"**Total Sentences:** {total_sentences}")
            st.markdown(f"**Positive Sentences:** {count_positive}")
            st.markdown(f"**Negative Sentences:** {count_negative}")
            st.markdown(f"**Neutral Sentences:** {count_neutral}")
            
            st.subheader("Sentiment Breakdown")
            fig1, ax1 = plt.subplots()
            labels = ['Positive', 'Negative', 'Neutral']
            sizes = [pos_percent, neg_percent, neu_percent]
            colors = ['green', 'red', 'gray']
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)
            
            st.subheader("Sentence Sentiment Scores")
            df = pd.DataFrame(sentence_data)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            bar_colors = df["Compound Score"].apply(lambda x: "green" if x >= 0.05 else ("red" if x <= -0.05 else "gray"))
            ax2.bar(range(len(df)), df["Compound Score"], color=bar_colors)
            ax2.set_xlabel("Sentence Index")
            ax2.set_ylabel("Compound Score")
            ax2.set_title("Compound Sentiment Scores per Sentence")
            st.pyplot(fig2)
            
            st.subheader("Sentiment Distribution")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.hist(df["Compound Score"], bins=10, color='skyblue', edgecolor='black')
            ax3.set_xlabel("Compound Score")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Distribution of Sentence Compound Scores")
            st.pyplot(fig3)
            
            st.subheader("Detailed Sentence Analysis")
            filter_option = st.selectbox("Filter by Classification", ["All", "Positive", "Negative", "Neutral"])
            df_filtered = df if filter_option == "All" else df[df["Classification"] == filter_option]
            st.dataframe(df_filtered)
            
            st.subheader("Text with Sentiment Highlights")
            highlighted_html = highlight_text(sentence_data)
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            st.subheader("Download Analysis Report")
            report_content = f"Overall Sentiment: {sentiment} (Score: {overall_score:.2f})\n\n"
            report_content += "Sentiment Breakdown:\n"
            report_content += f"- Positive: {pos_percent:.1f}% ({count_positive} sentences)\n"
            report_content += f"- Negative: {neg_percent:.1f}% ({count_negative} sentences)\n"
            report_content += f"- Neutral: {neu_percent:.1f}% ({count_neutral} sentences)\n\n"
            report_content += "Detailed Sentence Analysis:\n"
            for data in sentence_data:
                report_content += f"Sentence: {data['Sentence']}\nScore: {data['Compound Score']:.2f} - {data['Classification']}\n\n"
            report_file = io.BytesIO(report_content.encode('utf-8'))
            st.download_button(label="📥 Download TXT Report", data=report_file, file_name="sentiment_analysis.txt", mime="text/plain")
            csv_file = io.BytesIO(df.to_csv(index=False).encode('utf-8'))
            st.download_button(label="📥 Download CSV Report", data=csv_file, file_name="sentiment_analysis.csv", mime="text/csv")
        else:
            st.warning("⚠️ Please enter or upload some text.")

#############################
# Comparison Analysis Mode
#############################
else:
    st.subheader("Comparison Analysis: Analyze Two Texts Side by Side")
    col1, col2 = st.columns(2)
    
    sample_text_a = (
        "The movie was a delightful experience with stunning visuals and an engaging storyline. Highly recommended!"
    )
    sample_text_b = (
        "The service was terrible and the food was bland. I would not recommend dining here."
    )
    
    with col1:
        st.markdown("### Text A")
        text_a = st.text_area("Enter first text for analysis:", height=150, value=sample_text_a, key="text_a")
    with col2:
        st.markdown("### Text B")
        text_b = st.text_area("Enter second text for analysis:", height=150, value=sample_text_b, key="text_b")
    
    if st.button("🔍 Analyze Both Texts"):
        if text_a and text_b:
            with st.spinner("Analyzing texts..."):
                time.sleep(1)
                result_a = analyze_sentiment(text_a)
                result_b = analyze_sentiment(text_b)
                
                overall_a = result_a["overall"]
                sentiment_a = "Positive" if overall_a >= 0.05 else "Negative" if overall_a <= -0.05 else "Neutral"
                df_a = pd.DataFrame(result_a["sentence_data"])
                total_a = len(result_a["sentence_data"])
                pos_a = len(result_a["pos_sentences"])
                neg_a = len(result_a["neg_sentences"])
                neu_a = total_a - pos_a - neg_a
                
                overall_b = result_b["overall"]
                sentiment_b = "Positive" if overall_b >= 0.05 else "Negative" if overall_b <= -0.05 else "Neutral"
                df_b = pd.DataFrame(result_b["sentence_data"])
                total_b = len(result_b["sentence_data"])
                pos_b = len(result_b["pos_sentences"])
                neg_b = len(result_b["neg_sentences"])
                neu_b = total_b - pos_b - neg_b
                
                st.success("Analysis Completed!")
            
            st.markdown("### Overall Summary")
            summary_df = pd.DataFrame({
                "": ["Overall Score", "Sentiment", "Total Sentences", "Positive", "Negative", "Neutral"],
                "Text A": [f"{overall_a:.2f}", sentiment_a, total_a, pos_a, neg_a, neu_a],
                "Text B": [f"{overall_b:.2f}", sentiment_b, total_b, pos_b, neg_b, neu_b]
            })
            st.dataframe(summary_df)
            
            st.subheader("Comparative Overall Sentiment")
            fig_comp, ax_comp = plt.subplots()
            texts = ["Text A", "Text B"]
            overall_scores = [overall_a, overall_b]
            bar_colors = ["green" if s >= 0.05 else "red" if s <= -0.05 else "gray" for s in overall_scores]
            ax_comp.bar(texts, overall_scores, color=bar_colors)
            ax_comp.set_ylabel("Overall Compound Score")
            st.pyplot(fig_comp)
            
            st.markdown("### Detailed Sentence Analysis for Text A")
            filter_option_a = st.selectbox("Filter Text A", ["All", "Positive", "Negative", "Neutral"], key="filter_a")
            df_a_filtered = df_a if filter_option_a == "All" else df_a[df_a["Classification"] == filter_option_a]
            st.dataframe(df_a_filtered)
            
            st.markdown("### Detailed Sentence Analysis for Text B")
            filter_option_b = st.selectbox("Filter Text B", ["All", "Positive", "Negative", "Neutral"], key="filter_b")
            df_b_filtered = df_b if filter_option_b == "All" else df_b[df_b["Classification"] == filter_option_b]
            st.dataframe(df_b_filtered)
            
            st.markdown("### Text with Sentiment Highlights")
            st.markdown("**Text A:**")
            st.markdown(highlight_text(result_a["sentence_data"]), unsafe_allow_html=True)
            st.markdown("**Text B:**")
            st.markdown(highlight_text(result_b["sentence_data"]), unsafe_allow_html=True)
            
            st.subheader("Download Combined Analysis Report")
            report_content = "=== Text A Analysis ===\n"
            report_content += f"Overall Sentiment: {sentiment_a} (Score: {overall_a:.2f})\n"
            report_content += f"Total Sentences: {total_a}\nPositive: {pos_a}\nNegative: {neg_a}\nNeutral: {neu_a}\n\n"
            report_content += "Detailed Sentence Analysis:\n"
            for data in result_a["sentence_data"]:
                report_content += f"Sentence: {data['Sentence']}\nScore: {data['Compound Score']:.2f} - {data['Classification']}\n\n"
            report_content += "\n=== Text B Analysis ===\n"
            report_content += f"Overall Sentiment: {sentiment_b} (Score: {overall_b:.2f})\n"
            report_content += f"Total Sentences: {total_b}\nPositive: {pos_b}\nNegative: {neg_b}\nNeutral: {neu_b}\n\n"
            report_content += "Detailed Sentence Analysis:\n"
            for data in result_b["sentence_data"]:
                report_content += f"Sentence: {data['Sentence']}\nScore: {data['Compound Score']:.2f} - {data['Classification']}\n\n"
            report_file = io.BytesIO(report_content.encode('utf-8'))
            st.download_button(label="📥 Download TXT Report", data=report_file, file_name="comparison_sentiment_analysis.txt", mime="text/plain")
        else:
            st.warning("⚠️ Please enter text for both Text A and Text B.")


