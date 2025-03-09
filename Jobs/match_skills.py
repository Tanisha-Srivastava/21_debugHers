import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job data
with open("preprocessed_jobs.json", "r", encoding="utf-8") as file:
    job_listings = json.load(file)

# Extract job skills
job_skills = [job["skills"] for job in job_listings]  

# Compute TF-IDF & similarity
vectorizer = TfidfVectorizer(stop_words="english")  
tfidf_matrix = vectorizer.fit_transform(job_skills)  

# Store skill similarity scores
skill_scores = {}

def find_matching_jobs(user_skills):
    """Finds the best job matches based on skill similarity using TF-IDF + Cosine Similarity"""

    user_tfidf = vectorizer.transform([user_skills])  
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)[0]  

    for i, job in enumerate(job_listings):
        skill_scores[job["title"]] = float(similarity_scores[i])  # Convert to JSON serializable

# Example Run (This should be called from an input script)
user_input_skills = "Cloud, NodeJS, Data Analytics"
find_matching_jobs(user_input_skills)

# Save skill similarity
with open("skill_similarity.json", "w", encoding="utf-8") as file:
    json.dump(skill_scores, file, indent=4)

print("✅ Skill Similarity saved as skill_similarity.json")
