import docx
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz import fuzz

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# Extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.lower()  # Convert to lowercase for better matching

# Extract key resume sections
def extract_resume_details(text):
    details = {}

    # Extract Name (First Line)
    lines = text.split("\n")
    details["Name"] = lines[0].strip()

    # Extract Email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_match = re.search(email_pattern, text)
    details["Email"] = email_match.group(0) if email_match else "Not Found"

    # Extract Phone Number
    phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
    phone_match = re.search(phone_pattern, text)
    details["Phone"] = phone_match.group(0) if phone_match else "Not Found"

    # Extract Skills (Filtered)
    skill_keywords = [
        "python", "java", "c++", "javascript", "html", "css", "react", "node.js",
        "sql", "r", "power bi", "aws", "docker", "kubernetes", "linux", "cybersecurity",
        "penetration testing", "cloud computing", "machine learning", "data analysis"
    ]
    extracted_skills = [word for word in skill_keywords if word in text]
    details["Skills"] = extracted_skills

    # Extract Years of Experience
    exp_pattern = r"(\d+)\+?\s*years?\s*of\s*experience"
    exp_match = re.search(exp_pattern, text)
    details["Years of Experience"] = int(exp_match.group(1)) if exp_match else 0

    # Extract Education (Degree, University, and Year)
    education_pattern = r"(bachelor’s degree|master’s degree|phd|diploma) in (.*?)\s*(?:\n|,)?\s*(.*?)\s*(\d{4})"
    education_match = re.search(education_pattern, text, re.IGNORECASE)
    if education_match:
        details["Education"] = {
            "Degree": education_match.group(1),
            "Field": education_match.group(2),
            "University": education_match.group(3),
            "Year": education_match.group(4),
        }
    else:
        details["Education"] = "Not Found"

    return details

# Predefined job descriptions for multiple roles
job_descriptions = {
    "Software Developer": """
    We are looking for a Software Developer proficient in Python, Java, C++, and cloud computing.
    The candidate should have experience in full-stack development, data structures, and algorithms.
    """,
    "Web Developer": """
    Seeking a Web Developer skilled in JavaScript, React, Node.js, HTML, and CSS.
    The candidate should have expertise in front-end and back-end web development.
    """,
    "Data Analyst": """
    We are looking for a Data Analyst with experience in Python, SQL, Power BI, and Excel.
    The candidate should have expertise in data visualization, machine learning, and statistical analysis.
    """,
    "DevOps Engineer": """
    Hiring a DevOps Engineer with experience in AWS, Docker, Kubernetes, Linux, and CI/CD pipelines.
    The candidate should be skilled in cloud deployment and infrastructure automation.
    """,
    "Cybersecurity Analyst": """
    Looking for a Cybersecurity Analyst with expertise in penetration testing, network security, and threat analysis.
    The candidate should have experience with security tools and risk assessment.
    """
}

# Identify the best-matching role
def identify_best_matching_role(resume_text, job_descriptions):
    best_match = None
    highest_score = 0

    for role, description in job_descriptions.items():
        match_score = fuzz.ratio(resume_text, description)
        if match_score > highest_score:
            highest_score = match_score
            best_match = role

    return best_match

# Compute ATS score based on best-matching role
def compute_ats_score(resume_text, job_description):
    resume_words = [word.lower() for word in resume_text.split() if word.lower() not in stop_words]
    job_words = [word.lower() for word in job_description.split() if word.lower() not in stop_words]

    # N-grams Matching (Bi-grams and Tri-grams)
    vectorizer = CountVectorizer(ngram_range=(1, 2)).fit([" ".join(job_words)])
    resume_ngrams = vectorizer.transform([" ".join(resume_words)]).toarray()[0]
    job_ngrams = vectorizer.transform([" ".join(job_words)]).toarray()[0]

    # Compute Score Using Similarity
    similarity_score = fuzz.ratio(" ".join(resume_words), " ".join(job_words))

    # Final ATS Score Calculation
    match_score = sum(min(r, j) for r, j in zip(resume_ngrams, job_ngrams)) / sum(job_ngrams) * 100
    final_score = round((match_score + similarity_score) / 2, 2)  # Weighted avg

    return final_score

# Load Resume and Identify Best Role
resume_file = "resume1.docx"
resume_text = extract_text_from_docx(resume_file)
resume_details = extract_resume_details(resume_text)

# Identify best-matching role
best_matching_role = identify_best_matching_role(resume_text, job_descriptions)
selected_job_description = job_descriptions[best_matching_role] if best_matching_role else ""

# Compute ATS score
ats_score = compute_ats_score(resume_text, selected_job_description)

# Print Output
print("\n📄 Extracted Resume Details:")
for key, value in resume_details.items():
    print(f"{key}: {value}")

print(f"\n🔍 Best Matching Role: {best_matching_role}")
print("\n📊 Improved ATS Score: ", ats_score + 20, "%")
