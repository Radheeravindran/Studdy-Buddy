import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer

# Load the model and tokenizer
@st.cache_resource
def load_summarizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

# Function to summarize text
def summarize_content(text, tokenizer, model):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app UI
def main():
    st.title("📚 Study Buddy - Text Summarizer")

    # Load the summarizer model
    tokenizer, model = load_summarizer()

    # User input section
    user_input = st.text_area("Enter text to summarize", placeholder="Type or paste your text here...")

    if st.button("Summarize"):
        if user_input.strip():
            with st.spinner("Summarizing..."):
                summary = summarize_content(user_input, tokenizer, model)
                st.success("Summary:")
                st.write(summary)
        else:
            st.warning("Please enter some text before summarizing.")

# Run the app
if __name__ == "__main__":
    main()
