from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Load the dataset
product_data = pd.read_csv('product_data.csv')

# Check if required columns are present
required_columns = ['brandName', 'category', 'description', 'recyclable', 'price', 'sustainabilityRating', 'carbonfootprint', 'productName']
for col in required_columns:
    if col not in product_data.columns:
        raise ValueError(f"Missing required column: {col}")

# Preprocess data
product_data['combined_text'] = (
    product_data['brandName'].fillna('') + ' ' +
    product_data['category'].fillna('') + ' ' +
    product_data['description'].fillna('') + ' ' +
    product_data['recyclable'].fillna('')
)

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


# Scale numerical features


# Combine text and numerical features
combined_features = hstack([text_features, numerical_features])

# Compute similarity matrix
cosine_sim = cosine_similarity(combined_features, combined_features)

# Define recommendation function
def get_recommendations(product_name, top_n=5):
    try:
        # Get index of product
        idx = product_data[product_data['productName'].str.contains(product_name, case=False)].index[0]
    except IndexError:
        return None

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    product_indices = [i[0] for i in sim_scores]

    # Retrieve recommended products
    recommendations = product_data.iloc[product_indices][['productName', 'brandName', 'category', 'sellingPrice', 'sustainabilityRating', 'carbonfootprint', 'recyclable']]
    return recommendations.to_dict(orient='records')

# Set up Flask API
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    product_name = request.args.get('product_name')
    if not product_name:
        return jsonify({'error': 'Product name is required'}), 400
    
    recommendations = get_recommendations(product_name)
    if recommendations is None:
        return jsonify({'error': 'Product not found'}), 404
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

