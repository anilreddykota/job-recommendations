from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pandas as pd
import requests

from bson import ObjectId
from flask_cors import CORS
app = Flask(__name__)

CORS(app)

# MongoDB connection
client = MongoClient('mongodb+srv://vignaramtejtelagarapu:vzNsqoKpAzHRdN9B@amile.auexv.mongodb.net/?retryWrites=true&w=majority&appName=Amile')
db = client['amile']
job_collection = db['internships']

job_df = None
tfidf_matrix = None
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

def clean_text(text):
    return ' '.join(str(text).strip().lower().split())

def fetch_jobs():
    jobs = job_collection.find()
    return list(jobs)

def combine_text_fields(row):
    fields = ['role', 'companyName', 'skillsRequired', 'responsibilities', 'qualifications']
    return ' '.join(clean_text(row.get(field, '')) for field in fields)

def train_model():
    global job_df, tfidf_matrix
    jobs = fetch_jobs()
    job_df = pd.DataFrame(jobs)

    job_df['_id'] = job_df['_id'].apply(lambda x: str(x))

    print(f"Number of jobs: {len(job_df)}")

    job_df['combined_text'] = job_df.apply(combine_text_fields, axis=1)

    print(f"Sample combined text: {job_df['combined_text'].iloc[0] if len(job_df) > 0 else 'No data'}")

    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['combined_text'])

def recommend_jobs(user_skills, top_n=5):
    global job_df, tfidf_matrix
    if job_df is None or tfidf_matrix is None:
        train_model()

    cleaned_skills = clean_text(user_skills)
    user_vector = tfidf_vectorizer.transform([cleaned_skills])

    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    columns_to_return = [
        '_id', 'role', 'companyName', 'stipend', 'hours', 'type', 'modeOfWork',
        'location', 'startDate', 'endDate', 'skillsRequired', 'responsibilities',
        'qualifications', 'applicationDeadline', 'contactEmail', 'website',
        'benefits', 'postedAt', 'isActive'
    ]
    recommended_jobs = job_df.iloc[top_indices][columns_to_return]
    return recommended_jobs

scheduler = BackgroundScheduler()
scheduler.add_job(train_model, IntervalTrigger(minutes=30))  # Train the model every 30 minutes
scheduler.start()

train_model()
# @app.route('/scrape',methods=['GET'])
# def scrape():

#         url = "https://linkedin-data-scraper.p.rapidapi.com/company_insights"

#         querystring = {"link":"https://www.linkedin.com/company/google"}

#         headers = {
# 	        "x-rapidapi-key": "39ee2ff0femsh87c2169491d5dfbp17049fjsne81b437ca161",
# 	    "x-rapidapi-host": "linkedin-data-scraper.p.rapidapi.com"
#         }

#         response = requests.get(url, headers=headers, params=querystring)
#         return response.json()
    
@app.route('/recommend', methods=['POST'])
def recommend():
    user_skills = request.json.get('skills')
    if not user_skills:
        return jsonify({"error": "Skills are required"}), 400

    recommended_jobs = recommend_jobs(user_skills)
    return jsonify(recommended_jobs.to_dict(orient='records'))

@app.route('/train', methods=['POST'])
def train():
    train_model()
    return jsonify({"message": "Model trained successfully"})

if __name__ == "__main__":
    print("Job Recommendation System is running. Press Ctrl+C to exit.")
    try:
         port = int(os.environ.get("PORT", 5000))
         app.run(host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        scheduler.shutdown()
        client.close()
