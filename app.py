import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['amile']
job_collection = db['internships']

# Global variables
job_df = None
tfidf_matrix = None
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Data Preprocessing
def clean_text(text):
    return ' '.join(str(text).strip().lower().split())

# Fetch job data from MongoDB
def fetch_jobs():
    jobs = job_collection.find()
    return list(jobs)

# Combine text fields
def combine_text_fields(row):
    fields = ['role', 'companyName', 'skillsRequired', 'responsibilities', 'qualifications']
    return ' '.join(clean_text(row.get(field, '')) for field in fields)

# Train the model with live data
def train_model():
    global job_df, tfidf_matrix
    jobs = fetch_jobs()
    job_df = pd.DataFrame(jobs)
    
    print(f"Number of jobs: {len(job_df)}")
    
    job_df['combined_text'] = job_df.apply(combine_text_fields, axis=1)
    
    print(f"Sample combined text: {job_df['combined_text'].iloc[0] if len(job_df) > 0 else 'No data'}")
    
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['combined_text'])

# Recommendation function
def recommend_jobs(user_skills, top_n=5):
    global job_df, tfidf_matrix
    if job_df is None or tfidf_matrix is None:
        train_model()
    
    cleaned_skills = clean_text(user_skills)
    user_vector = tfidf_vectorizer.transform([cleaned_skills])
    
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    columns_to_return = ['role', 'companyName', 'skillsRequired', 'responsibilities', 'qualifications']
    recommended_jobs = job_df.iloc[top_indices][columns_to_return]
    return recommended_jobs

# Set up APScheduler to periodically update the model
scheduler = BackgroundScheduler()
scheduler.add_job(train_model, IntervalTrigger(minutes=30))  # Train the model every 30 minutes
scheduler.start()

# Initial model training
train_model()

# Main loop
if __name__ == "__main__":
    print("Job Recommendation System is running. Press Ctrl+C to exit.")
    try:
        while True:
            user_skills = input("Enter your skills (or 'q' to quit): ")
            if user_skills.lower() == 'q':
                break
            recommended_jobs = recommend_jobs(user_skills)
            print("\nRecommended Jobs:")
            print(recommended_jobs)
            print("\n")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        scheduler.shutdown()
        client.close()