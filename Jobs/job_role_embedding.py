from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

# Load preprocessed job data
with open("preprocessed_jobs.json", "r", encoding="utf-8") as file:
    job_listings = json.load(file)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract unique job roles & compute embeddings
unique_job_roles = list(set(job["job_role"] for job in job_listings if job["job_role"]))
job_role_embeddings = model.encode(unique_job_roles, convert_to_tensor=True)

# Compute similarity for each job
job_role_scores = {}

for job in job_listings:
    job_title = job["title"]
    job_role = job["job_role"]

    # Compute embedding & similarity
    user_embedding = model.encode(job_role, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_embedding, job_role_embeddings)[0].cpu().numpy()

    # Get best match
    best_match_idx = np.argmax(similarity_scores)
    best_match = unique_job_roles[best_match_idx]
    best_score = similarity_scores[best_match_idx]

    job_role_scores[job_title] = float(best_score)  # Convert to JSON serializable

# Save job role similarity scores
with open("job_role_similarity.json", "w", encoding="utf-8") as file:
    json.dump(job_role_scores, file, indent=4)

print("✅ Job Role Similarity saved as job_role_similarity.json")
