import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# Loading the saved Logistic Regression model
model = joblib.load('best_model.pkl')
# Loading the TF-IDF vectorizer used during model training
vectorizer = joblib.load('tfidf_vectorizer.pkl')
# Downloading NLTK data (if not already downloaded)
nltk.download('stopwords')

# Defining a function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Converting to lowercase
    text = re.sub(r'\d+', '', text)  # Removing numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
    text = text.strip()  # Removing leading/trailing whitespace
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Removing stopwords
    return text

# Setting page config for wider layout
st.set_page_config(layout="wide")

# Setting Sidebar for theme selection
st.sidebar.title("App Settings")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Blue"])

# Applying the user's selected theme
if theme == "Blue":
    st.markdown("""
    <style>
    .stApp {
        background-color: #E6F2FF;
        color: #0A2F5C;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app interface
st.title('Sentiment Classifier Tool 🤖')

# Input text box for user input
user_input = st.text_area("Enter your sentiment:")

# Customizing CSS for the green button
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

def simulate_progress(progress, status_text, start, end, duration, step_text):
    for i in range(start, end + 1):
        progress.progress(i)
        status_text.text(f"{step_text}... {i}%")
        time.sleep(duration / (end - start))

def get_recommendation(sentiment):
    if sentiment == "Positive":
        return [
            "Great job! Keep up the positive attitude! 👍",
            "Consider sharing your positive experience with others. 😊",
            "Use this positive energy to tackle any challenges ahead. 💪"
        ]
    else:
        return [
            "Remember, every cloud has a silver lining. Stay hopeful! 🌈",
            "Consider talking to someone you trust about your feelings. 🗣️",
            "Take some time for self-care activities you enjoy. 🧘‍♀️"
        ]

if st.button("Analyze Sentiment"):
    if user_input:
        # Creating a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Preprocessing the user input
        simulate_progress(progress_bar, status_text, 0, 33, 1, "Preprocessing text 📝")
        processed_input = preprocess_text(user_input)

        # Vectorizing the input text
        simulate_progress(progress_bar, status_text, 34, 66, 1, "Vectorizing input 🔢")
        input_vector = vectorizer.transform([processed_input])

        # Predicting the sentiment
        simulate_progress(progress_bar, status_text, 67, 99, 1, "Analyzing sentiment 🔍")
        prediction = model.predict(input_vector)[0]

        # Mapping the prediction to sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"
        progress_bar.progress(100)
        status_text.text("Analysis complete! 100% ✅")
        time.sleep(0.5)  # Short pause before showing result

        # Clearing the progress bar and status text
        progress_bar.empty()
        status_text.empty()

        # Displaying the result with emoji
        emoji = "😃" if sentiment == "Positive" else "😔"
        st.write(f"The sentiment is: **{sentiment}** {emoji}")

        # Displaying recommendations
        st.subheader("Recommendations based on your sentiment:")
        recommendations = get_recommendation(sentiment)
        for rec in recommendations:
            st.write(f"• {rec}")

    else:
        st.write("Please enter some text to analyze. 📝")