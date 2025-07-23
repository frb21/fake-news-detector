import streamlit as st
import pickle
import re
import os
from datetime import datetime
import time

    # Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #1f2937;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2.5rem;
        line-height: 1.5;
    }
    
    .fake-news {
        background: #fee2e2;
        color: #dc2626;
        padding: 1.25rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
        border: 1px solid #fecaca;
    }
    
    .real-news {
        background: #dcfce7;
        color: #16a34a;
        padding: 1.25rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
        border: 1px solid #bbf7d0;
    }
    
    .info-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .warning-card {
        background: #fefce8;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #fde047;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea > div > div > textarea {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.6;
        border-radius: 8px;
        border: 1px solid #d1d5db;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
        background: #3b82f6;
        color: white;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }
    
    .stSelectbox > div > div {
        font-family: 'Inter', sans-serif;
    }
    
    .stRadio > div {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
    
    .stMetric {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and vectorizer"""
    try:
        # Load saved model
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        
        # Load saved vectorizer
        with open("vectorizer.pkl", "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        
        return model, vectorizer, True
    except FileNotFoundError:
        return None, None, False

# Function to preprocess text
def clean_text(text):
    """Clean and preprocess the input text"""
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text.strip()

# Function to predict fake news
def predict_fake_news(news, model, vectorizer):
    """Predict if news is fake or real"""
    cleaned_news = clean_text(news)
    vectorized_news = vectorizer.transform([cleaned_news])
    prediction = model.predict(vectorized_news)
    
    result = "Fake News" if prediction[0] == 'FAKE' else "Real News"
    
    return result, cleaned_news

def main():
    # Header
    st.markdown('<h1 class="main-header">Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyze news articles to determine their authenticity using machine learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, vectorizer, model_loaded = load_model_and_vectorizer()
    
    if not model_loaded:
        st.error("Model files not found! Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory as this app.")
        st.markdown("""
        <div class="warning-card">
            <h4>Required Files:</h4>
            <ul>
                <li><code>model.pkl</code> - Your trained machine learning model</li>
                <li><code>vectorizer.pkl</code> - Your text vectorizer</li>
                <li><code>app.py</code> - This Streamlit application</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Success message for model loading
    st.success("Model loaded successfully")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter News Article")
        
        # Text input options
        input_method = st.radio(
            "Choose input method:",
            ["Type/Paste Article", "Example Articles"],
            horizontal=True
        )
        
        if input_method == "Type/Paste Article":
            news_text = st.text_area(
                "Paste your news article here:",
                height=300,
                placeholder="Enter the news article you want to check for authenticity...",
                help="Paste the full text of the news article you want to analyze."
            )
        else:
            # Example articles for testing
            examples = {
                "Real News Example": "Scientists at Stanford University have developed a new method for detecting early-stage cancer using artificial intelligence. The research, published in Nature Medicine, shows promising results in clinical trials with 95% accuracy in identifying malignant tumors. The team used machine learning algorithms trained on thousands of medical images to create this breakthrough diagnostic tool.",
                "Fake News Example": "BREAKING: Local man discovers that drinking lemon water every morning for 30 days will cure all diseases including cancer and diabetes. Doctors hate this one simple trick that pharmaceutical companies don't want you to know. Share this post to save lives and spread the truth that mainstream media is hiding from you."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            news_text = st.text_area(
                "Example article:",
                value=examples[selected_example],
                height=200,
                help="This is an example article for testing purposes."
            )
    
    with col2:
        st.markdown("### How it works")
        st.markdown("""
        <div class="info-card">
            <h4>Analysis Process:</h4>
            <ol>
                <li><strong>Text Preprocessing:</strong> Cleans and standardizes the input text</li>
                <li><strong>Feature Extraction:</strong> Converts text into numerical features</li>
                <li><strong>ML Prediction:</strong> Uses trained model to classify the news</li>
                <li><strong>Result Display:</strong> Shows whether the article is real or fake</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Model Info")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Status", "Ready", delta="Loaded")
        with col_b:
            st.metric("Accuracy", "85-95%", delta="Estimated")
    
    # Prediction section
    if st.button("Analyze Article", type="primary", use_container_width=True):
        if not news_text.strip():
            st.warning("Please enter a news article to analyze.")
            return
        
        # Show loading spinner
        with st.spinner("Analyzing article..."):
            time.sleep(1)  # Small delay for better UX
            
            try:
                # Make prediction
                result, cleaned_text = predict_fake_news(news_text, model, vectorizer)
                
                # Display results
                st.markdown("---")
                st.markdown("### Analysis Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if result == "Fake News":
                        st.markdown(f"""
                        <div class="fake-news">
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="real-news">
                            {result}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Prediction:**")
                    st.write(result)
                    st.markdown("**Status:**")
                    st.write("Analysis Complete")
                
                # Additional information
                st.markdown("### Analysis Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Text Length:**")
                    st.write(f"{len(news_text)} characters")
                    st.markdown("**Word Count:**")
                    st.write(f"{len(news_text.split())} words")
                
                with col2:
                    st.markdown("**Processed Text Length:**")
                    st.write(f"{len(cleaned_text)} characters")
                    st.markdown("**Analysis Time:**")
                    st.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()