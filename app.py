import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Load and preprocess the dataset
def load_data():
    product_data = pd.read_csv('product_data.csv')
    product_data['combined_text'] = (
        product_data['brandName'].fillna('') + ' ' +
        product_data['category'].fillna('') + ' ' +
        product_data['description'].fillna('') + ' ' +
        product_data['recyclable'].fillna('')
    )
    return product_data

product_data = load_data()

# Initialize TF-IDF vectorizer and transform text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
text_features = tfidf_vectorizer.fit_transform(product_data['combined_text'])

# Scale numerical features
product_data['price'] = pd.to_numeric(product_data['price'], errors='coerce').fillna(0)
product_data['sustainabilityRating'] = pd.to_numeric(product_data['sustainabilityRating'], errors='coerce').fillna(0)
product_data['carbonfootprint'] = pd.to_numeric(product_data['carbonfootprint'], errors='coerce').fillna(0)

# Normalize numerical features
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(product_data[['price', 'sustainabilityRating', 'carbonfootprint']])

# Combine text and numerical features
combined_features = hstack([text_features, numerical_features])

# Compute similarity matrix
cosine_sim = cosine_similarity(combined_features, combined_features)

# Recommendation function
def get_recommendations(product_name, top_n=5):
    try:
        idx = product_data[product_data['productName'].str.contains(product_name, case=False)].index[0]
    except IndexError:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]

    recommendations = product_data.iloc[product_indices][['productName', 'brandName', 'category', 'sellingPrice', 'sustainabilityRating', 'carbonfootprint', 'recyclable', 'image_url']]
    return recommendations

# Streamlit app UI
st.title("Product Recommendation System")
st.write("Enter a product name to get similar product recommendations")

# Input box for product name
product_name = st.text_input("Product Name", "")

# Button to get recommendations
if st.button("Get Recommendations"):
    if product_name:
        recommendations = get_recommendations(product_name)
        if recommendations is not None:
            st.write("Top Recommendations:")
            for index, row in recommendations.iterrows():
                st.image(row['image_url'], width=150)
                st.write(f"**Product Name:** {row['productName']}")
                st.write(f"**Brand Name:** {row['brandName']}")
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Selling Price:** {row['sellingPrice']}")
                st.write(f"**Sustainability Rating:** {row['sustainabilityRating']}")
                st.write(f"**Carbon Footprint:** {row['carbonfootprint']}")
                st.write(f"**Recyclable:** {row['recyclable']}")
                st.write("---")
        else:
            st.write("No recommendations found. Please try a different product name.")
    else:
        st.write("Please enter a product name.")

