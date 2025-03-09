import json
import re
from rapidfuzz import fuzz, process

# 1️⃣ Load JSON Data
with open("final_jobs_corrected.json", "r", encoding="utf-8") as file:
    job_listings = json.load(file)

# 2️⃣ Normalize job roles (lowercasing, stripping spaces)
for job in job_listings:
    if job.get("job_role"):  # Ensures job_role exists and is not None
        job["job_role"] = job["job_role"].strip().lower()
    else:
        job["job_role"] = ""  # Assign an empty string if missing


# 3️⃣ Extract unique job roles
unique_job_roles = list(set(job["job_role"] for job in job_listings))

def find_best_match(user_input):
    """Find the closest job role match using fuzzy matching."""
    user_input = user_input.strip().lower()  # Normalize input
    
    match, score = process.extractOne(user_input, unique_job_roles, scorer=fuzz.partial_ratio)
    return match if score > 70 else None  # Only return if similarity > 70%

# 4️⃣ Preprocess Experience Field
def extract_experience_range(exp_string):
    """Extracts min and max years of experience from string (e.g., '1 to 5 years' → (1, 5))"""
    numbers = list(map(int, re.findall(r'\d+', exp_string)))  # Extract numbers
    return (numbers[0], numbers[-1]) if numbers else (0, 0)  # Default (0,0) if no numbers

# 5️⃣ Normalize Other Fields (Daycare & Location)
for job in job_listings:
    job["experience_range"] = extract_experience_range(job["experience"])
    job["daycare"] = 1 if job["daycare"].strip().lower() == "yes" else 0  # Convert to 1/0
    job["location"] = job["location"].strip().lower()  # Normalize location

# Save preprocessed data (Optional)
with open("preprocessed_jobs.json", "w", encoding="utf-8") as file:
    json.dump(job_listings, file, indent=4)

print("✅ Preprocessing Complete! Preprocessed data saved.")
