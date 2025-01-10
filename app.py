import streamlit as st
import pickle
import streamlit.components.v1 as components

# Load the model and vectorizer
try:
    model = pickle.load(open('spam.pkl', 'rb'))
    cv = pickle.load(open('vec.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please upload 'spam.pkl' and 'vec.pkl'.")

# Custom HTML meta tags for social media sharing
meta_tags = """
<meta property="og:title" content="MailGuard AI: Your Spam Defense" />
<meta property="og:description" content="AI-powered email spam detection to safeguard your inbox from unwanted emails." />
<meta property="og:url" content="https://mailguard.streamlit.app/" />
"""

# Inject the meta tags into the HTML of your Streamlit app
components.html(meta_tags, height=0)

# Main function for the Streamlit app
def main():
    # Set up the page title and description
    st.set_page_config(page_title="MailGuard AI", page_icon="📬", layout="wide")

    # Custom page styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title and introduction
    st.title("📬 MailGuard AI: Spam Email Classifier")
    st.markdown(
        """
        ### Safeguard your inbox with AI-powered email classification:
        - **Spam**: Unwanted or junk email.
        - **Ham**: Legitimate or useful email.
        """
    )

    # Input text area for email content
    st.subheader("🚀 Classify Your Email")
    user_input = st.text_area("Enter the email content you want to classify:", height=150)

    # Process and classify the input on button click
    if st.button("🔍 Classify Email"):
        if user_input.strip():
            try:
                # Prepare input for model
                data = [user_input]
                vec = cv.transform(data).toarray()

                # Predict the class
                result = model.predict(vec)
                confidence = model.predict_proba(vec).max() * 100

                # Display results with enhanced UI
                st.write("### 📨 Input Email:")
                st.markdown(f"```{user_input}```")
                if result[0] == 0:
                    st.success(f"✅ This is Not a Spam Email ({confidence:.2f}% confidence).")
                else:
                    st.error(f"🚫 This is a Spam Email ({confidence:.2f}% confidence).")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("⚠️ Please enter email content to classify.")

    # Footer section
    st.markdown("---")
    st.markdown(
        """
        Developed with ❤️ using Streamlit | [GitHub Repository](https://github.com/YourRepo/MailGuardAI)
        """)

# Run the main function
if __name__ == "__main__":
    main()
