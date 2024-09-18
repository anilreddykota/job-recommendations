from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from bson import ObjectId
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# MongoDB connection
client = MongoClient('mongodb+srv://vignaramtejtelagarapu:vzNsqoKpAzHRdN9B@amile.auexv.mongodb.net/?retryWrites=true&w=majority&appName=Amile')
db = client['test']
job_collection = db['internships']

job_df = None
tfidf_matrix = None
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Helper function to clean text
def clean_text(text):
    return ' '.join(str(text).strip().lower().split())

# Fetch jobs from the MongoDB collection
def fetch_jobs():
    jobs = job_collection.find()
    return list(jobs)

# Combine text fields for vectorization
def combine_text_fields(row):
    fields = ['role', 'companyName', 'skillsRequired', 'responsibilities', 'qualifications']
    return ' '.join(clean_text(row.get(field, '')) for field in fields)

# Train the model to create job vector representations
def train_model():
    global job_df, tfidf_matrix

    # Fetch jobs from the database
    jobs = fetch_jobs()
    if not jobs:
        raise ValueError("No job data available to train the model.")

    job_df = pd.DataFrame(jobs)

    # Ensure _id column exists
    if '_id' not in job_df.columns:
        raise KeyError("_id field is missing in the fetched job data.")

    # Convert MongoDB ObjectId to string
    job_df['_id'] = job_df['_id'].apply(lambda x: str(x))

    print(f"Number of jobs: {len(job_df)}")

    # Combine text fields for vectorization
    job_df['combined_text'] = job_df.apply(combine_text_fields, axis=1)

    if len(job_df) > 0:
        print(f"Sample combined text: {job_df['combined_text'].iloc[0]}")
    else:
        print("No data available in the DataFrame.")

    # Fit TF-IDF vectorizer and create job matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['combined_text'])

# Recommend jobs based on user skills
def recommend_jobs(user_skills, top_n=5):
    global job_df, tfidf_matrix

    if job_df is None or tfidf_matrix is None:
        train_model()

    cleaned_skills = clean_text(user_skills)
    user_vector = tfidf_vectorizer.transform([cleaned_skills])

    # Calculate cosine similarity between user skills and jobs
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]  # Top N most similar jobs

    # Columns to return
    columns_to_return = [
        '_id', 'role', 'companyName', 'stipend', 'hours', 'type', 'modeOfWork',
        'location', 'startDate', 'endDate', 'skillsRequired', 'responsibilities',
        'qualifications', 'applicationDeadline', 'contactEmail', 'website',
        'benefits', 'postedAt', 'isActive'
    ]
    recommended_jobs = job_df.iloc[top_indices][columns_to_return]
    return recommended_jobs

# Route to recommend jobs
@app.route('/recommend', methods=['POST'])
def recommend():
    user_skills = request.json.get('skills')
    if not user_skills:
        return jsonify({"error": "Skills are required"}), 400

    try:
        recommended_jobs = recommend_jobs(user_skills)
        return jsonify(recommended_jobs.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to train the model
@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Job Recommendation System is running. Press Ctrl+C to exit.")
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(debug=True,port=port)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        client.close()
