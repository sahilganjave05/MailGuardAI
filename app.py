import streamlit as st
import pickle
import streamlit.components.v1 as components

# Set the page configuration
st.set_page_config(page_title="MailGuard AI", page_icon="📬", layout="wide", initial_sidebar_state="expanded")

# Load the model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vec.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please upload 'spam.pkl' and 'vec.pkl'.")

# Custom HTML meta tags for social media sharing
meta_tags = """
<meta property="og:title" content="MailGuard AI: Spam Email Classifier" />
<meta property="og:description" content="Classify your emails as Spam or Not Spam using AI. Protect your inbox from unwanted emails." />
<meta property="og:url" content="https://mailguard.streamlit.app/" />
"""
components.html(meta_tags, height=0)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About MailGuard AI")
st.sidebar.markdown("""
MailGuard AI is a powerful machine learning application designed to help you:
- **Identify Spam Emails**.
- **Filter out unwanted content**.
- **Protect your inbox** from phishing and junk emails.
""")
st.sidebar.info("🔒 **Privacy**: Your data is not stored or shared.")

# Main function
def main():
    st.title("📧 MailGuard AI - Spam Email Classifier")
    st.markdown("""
        Welcome to **MailGuard AI**, a tool that uses **Machine Learning** to classify emails as either **Spam** or **Not Spam**.
    """)
    
    # Removed the image here
    # st.image("https://via.placeholder.com/1000x300?text=MailGuard+AI", use_column_width=True)

    # Input email content
    st.subheader("🔍 Email Classification")
    user_input = st.text_area("✏️ Enter the email content below:", height=200)

    # Classification result
    if st.button("🚀 Classify Email"):
        if user_input.strip():
            try:
                # Prepare input for model
                data = [user_input]
                vec = cv.transform(data).toarray()
                
                # Predict the class
                result = model.predict(vec)
                confidence = model.predict_proba(vec).max() * 100
                
                # Display results
                st.markdown("### 📤 Email Content:")
                st.write(f"```\n{user_input}\n```")
                
                st.markdown("### 📊 Classification Result:")
                if result[0] == 0:
                    st.success(f"✅ This is a **Not Spam** email ({confidence:.2f}% confidence).")
                else:
                    st.error(f"🚫 This is a **Spam** email ({confidence:.2f}% confidence).")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter some email content to classify.")

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using **Streamlit**")

# Run the main function
if __name__ == "__main__":
    main()
