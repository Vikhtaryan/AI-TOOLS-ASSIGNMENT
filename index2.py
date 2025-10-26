import spacy

# Load spaCy's English model with NER
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews (replace with actual data loaded from Kaggle)
reviews = [
    "I love my new Samsung Galaxy phone! The camera quality is amazing.",
    "The Apple MacBook battery life is disappointing and slow.",
    "Sony headphones have great sound but the build quality feels cheap.",
    "This Philips air fryer works perfectly and is super easy to use!",
    "The Amazon Basics charger stopped working in a week. Very unhappy."
]

# Define simple positive and negative sentiment keywords
positive_keywords = {"love", "great", "amazing", "perfectly", "easy", "good", "happy", "excellent"}
negative_keywords = {"disappointing", "slow", "cheap", "stopped", "unhappy", "bad", "poor"}

# Process each review
for review in reviews:
    doc = nlp(review)
    # Extract product and brand entities (try ORG, PRODUCT, etc.)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT"}]

    # Simple rule-based sentiment analysis
    review_tokens = set([token.text.lower() for token in doc])
    if review_tokens.intersection(positive_keywords):
        sentiment = "Positive"
    elif review_tokens.intersection(negative_keywords):
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Print results
    print(f"Review: {review}")
    print(f"Extracted Entities (Brands/Products): {entities}")
    print(f"Sentiment: {sentiment}\n")
