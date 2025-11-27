"""
IMDB Movie Review Classifier by Schutte

This Streamlit app:
- Loads a trained RNN model (imdb_rnn_model.h5)
- Loads the tokenizer (imdb_tokenizer.pkl)
- Lets the user type up to 5 movie reviews
- Predicts whether each review is positive or negative
"""

import json
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------------
# 1. Page configuration (title, layout, etc.)
# --------------------------------------------------------
# This line sets some basic options for the Streamlit page.
st.set_page_config(
    page_title="IMDB Movie Review Classifier",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------------
# 2. Helper: load model + tokenizer + config (cached)
# --------------------------------------------------------
# Streamlit re-runs the script from top to bottom on every interaction.
# @st.cache_resource makes sure we only load the model/tokenizer once,
# and then reuse them (so the app stays fast).
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained Keras model, tokenizer, and config from disk."""
    # Load trained model (this file should be created in your training notebook).
    model = tf.keras.models.load_model("imdb_rnn_model.h5")

    # Load tokenizer (also saved in your training notebook using pickle).
    with open("imdb_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load config with maxlen (sequence length) and possibly other info.
    with open("imdb_config.json", "r") as f:
        config = json.load(f)

    maxlen = config.get("maxlen", 200)  # default to 200 if key missing

    return model, tokenizer, maxlen


# Actually load everything (will be cached after the first run).
model, tokenizer, maxlen = load_model_and_tokenizer()


# --------------------------------------------------------
# 3. Helper functions for preprocessing and prediction
# --------------------------------------------------------
def preprocess_reviews(reviews, maxlen):
    """
    Turn a list of raw text reviews into padded integer sequences
    that the RNN model can understand.
    """
    # 1) Convert raw text to sequences of word indices using the tokenizer.
    sequences = tokenizer.texts_to_sequences(reviews)

    # 2) Pad/truncate every sequence to the same length (maxlen).
    #    - padding="post": add zeros at the end if the sequence is too short.
    #    - truncating="post": cut off extra tokens at the end if too long.
    padded = pad_sequences(
        sequences,
        maxlen=maxlen,
        padding="post",
        truncating="post"
    )

    return padded


def predict_sentiments(reviews, threshold=0.5):
    """
    Given a list of text reviews, return:
    - probabilities of being positive
    - corresponding text labels ("Positive"/"Negative")
    using the trained RNN model.
    """
    # Preprocess the raw reviews into model-ready input.
    x = preprocess_reviews(reviews, maxlen)

    # Use the model to predict probabilities.
    # model.predict returns an array of shape (num_samples, 1).
    probs = model.predict(x)

    # Flatten to 1D array for convenience.
    probs = probs.reshape(-1)

    # Turn probabilities into human labels using the threshold.
    labels = np.where(probs >= threshold, "Positive", "Negative")

    return probs, labels


# --------------------------------------------------------
# 4. Sidebar controls (settings)
# --------------------------------------------------------
# The sidebar holds optional settings so the main page stays clean.
st.sidebar.title("Settings")

# Slider to choose the probability threshold for "Positive".
# Default is 0.5 (standard for binary classification).
threshold = st.sidebar.slider(
    "Positive threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.01,
    help="If the model's positive probability ≥ this value, we call the review 'Positive'."
)

# Checkbox to toggle showing raw probability values.
show_probs = st.sidebar.checkbox(
    "Show raw prediction probability",
    value=True
)


# --------------------------------------------------------
# 5. Main page title and description
# --------------------------------------------------------
st.title("IMDB Movie Review Classifier by Schutte 🎬")

st.write(
    """
This simple web app uses a **Recurrent Neural Network (RNN)** trained on the IMDB dataset  
to classify movie reviews as **positive** or **negative**.

- Type up to **five** movie reviews in the boxes below.
- Click **“Classify Reviews”** to see the model's predictions.
"""
)


# --------------------------------------------------------
# 6. Text areas for up to 5 user-entered reviews
# --------------------------------------------------------
st.markdown("### Type your movie reviews")

# Some optional default example texts to make the app feel alive on first run.
default_reviews = [
    "I absolutely loved this movie. The acting was great and the story was very touching.",
    "This film was terrible. The plot made no sense and I was bored the whole time.",
    "",
    "",
    ""
]

review_texts = []

for i in range(5):
    # Create a subheader for each review input.
    st.subheader(f"Review {i + 1}")

    # Text area where the user can type a review.
    # - label: shown above the box
    # - value: default text (can be overwritten by the user)
    # - height: controls how tall the text box is
    text = st.text_area(
        label=f"Type review {i + 1} here:",
        value=default_reviews[i],
        height=120,
        key=f"review_{i + 1}"
    )

    # Strip leading/trailing whitespace for safety.
    review_texts.append(text.strip())


# --------------------------------------------------------
# 7. Button to trigger predictions
# --------------------------------------------------------
if st.button("Classify Reviews"):
    # Filter out completely empty reviews so we don't send blank strings to the model.
    non_empty_indices = [i for i, txt in enumerate(review_texts) if txt]

    if len(non_empty_indices) == 0:
        # If user clicked without typing anything, gently warn them.
        st.warning("Please type at least one non-empty review before classifying.")
    else:
        # Collect only the non-empty reviews in the same order.
        reviews_to_predict = [review_texts[i] for i in non_empty_indices]

        # Get probabilities and labels from our helper function.
        probs, labels = predict_sentiments(reviews_to_predict, threshold=threshold)

        st.markdown("---")
        st.markdown("## Classification Results")

        # Loop over results and display them nicely.
        for idx_in_list, (prob, label) in enumerate(zip(probs, labels)):
            # Map back to the original review index (0–4).
            original_idx = non_empty_indices[idx_in_list]

            st.markdown(f"### Review {original_idx + 1}")

            # Show the original review text.
            st.write(review_texts[original_idx])

            # Show the predicted sentiment label in bold.
            st.write(f"**Prediction:** {label}")

            # Optionally show the raw probability.
            if show_probs:
                st.write(f"Model probability of being positive: `{prob:.3f}`")

            # Add a horizontal line between reviews for readability.
            st.markdown("---")
else:
    # Friendly hint when the app first loads and button is not pressed yet.
    st.info("Enter your reviews above and click **'Classify Reviews'** to see predictions.")
