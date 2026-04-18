import re
import os
import tempfile
import docx
import pdfplumber
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# --- FLASK SETUP ---
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- LOAD GEMINI API ---
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Supported Gemini model

# --- LOAD JOB DATASET ---
BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "job.csv")

df = pd.read_csv(file_path)
drop_cols = [
    "advertiserurl", "employmenttype_jobstatus", "jobid", "joblocation_address",
    "postdate", "shift", "site_name", "uniq_id"
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# --- TEXT PROCESSING ---
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except:
        return ""

def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except:
        return ""

def extract_sections(text):
    sections = {}
    headings = {
        "Summary": ["summary", "professional summary", "career summary", "objective", "about me"],
        "Experience": ["experience", "work experience", "professional experience", "employment history"],
        "Education": ["education", "qualifications", "academic background"],
        "Skills": ["skills", "technical skills", "key skills", "areas of expertise"],
        "Projects": ["projects", "personal projects", "portfolio"],
    }
    heading_map = {alias.lower(): name for name, aliases in headings.items() for alias in aliases}
    pattern = r"(?i)\b(" + "|".join(heading_map.keys()) + r")\b[:\n\r]*"
    matches = list(re.finditer(pattern, text))

    if not matches:
        sections['Full Text'] = text.strip()
        return sections

    for i, match in enumerate(matches):
        start = match.end()
        end = len(text) if i + 1 == len(matches) else matches[i + 1].start()
        normalized = heading_map.get(match.group(1).lower())
        sections[normalized] = text[start:end].strip()
    return sections

def calculate_overall_similarity(resume_text, job_description):
    if not resume_text or not job_description:
        return 0.0
    resume_text = preprocess_text(resume_text)
    job_description = preprocess_text(job_description)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    return round(similarity * 100, 2)

# --- Calibrated score function ---
def calibrated_score(tfidf_score, general_advice_text):
    """
    Calibrate TF-IDF score to match the AI advice.
    """
    advice_lower = general_advice_text.lower()

    # Very poor fit: keep very low
    if "very poor" in advice_lower or "not a good fit" in advice_lower:
        return min(tfidf_score, 5)

    # Poor fit: boost moderately
    elif "poor fit" in advice_lower:
        return min(tfidf_score * 2, 25)

    # Moderate fit: modest boost
    elif "moderate fit" in advice_lower or "somewhat suitable" in advice_lower:
        return min(tfidf_score * 1.8, 45)

    # Good fit: slight boost
    elif "good fit" in advice_lower or "excellent match" in advice_lower:
        return min(tfidf_score * 1.2, 100)

    # Default: modest boost
    else:
        return min(tfidf_score * 1.2, 50)

def process_resume(file_path):
    text = ""
    if file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        return {"error": "Unsupported file format."}

    if text:
        return {"sections": extract_sections(text), "full_text": text}
    return {"error": "Could not extract text from file."}

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files or request.files['resume'].filename == '':
        return jsonify({'error': 'No file uploaded', 'suggestions': []})

    file = request.files['resume']
    job_title = request.form.get('jobtitle')
    if not job_title:
        return jsonify({'error': 'No job title provided', 'suggestions': []})

    _, ext = os.path.splitext(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp)
        file_path = tmp.name

    resume_data = process_resume(file_path)
    os.unlink(file_path)
    if "error" in resume_data:
        return jsonify({'error': resume_data['error'], 'suggestions': []})

    resume_text = resume_data.get('full_text', '')
    job_description = df.loc[df['jobtitle'] == job_title, 'jobdescription'].head(1).squeeze()

    # --- Overall Score (TF-IDF + cosine similarity) ---
    overall_score = calculate_overall_similarity(resume_text, job_description)

    # --- Gemini AI prompt ---
    prompt = f"""
You are an expert ATS resume analyzer and career consultant.

Compare the following resume with the job description for '{job_title}'.

Instructions:
1. Calculate ONE overall ATS match score (0–100) showing alignment with the job.
2. Provide a brief General Advice section (4 sentences) explaining why the resume is a good or poor fit.
3. Provide Improvement Suggestions (exactly 4 bullet points) with actionable advice to improve the resume.
4. Provide a Job Role Match section stating the most suitable role for this resume based on the ATS analysis.
5. Each section must start on a new line, using the following headings exactly:

General Advice:
[4 sentences here]

Improvement Suggestions:
- Suggestion 1
- Suggestion 2
- Suggestion 3
- Suggestion 4

Job Role Match :
[Role here only, no score]

Resume Text:
{resume_text}

Job Description:
{job_description}
"""

    try:
        response = model.generate_content([prompt])
        suggestions = response.text if hasattr(response, 'text') else "No suggestions available."
    except Exception as e:
        suggestions = f"Error generating suggestions: {e}"

    # --- Parse AI advice to align score ---
    def parseResumeAnalysis(content):
        content = content.replace("\r\n", "\n").strip()
        generalAdvice = ''
        improvementSuggestions = []
        jobRoleMatch = ''

        generalRegex = r"General Advice:\n([\s\S]*?)(?=\nImprovement Suggestions:|$)"
        suggestionsRegex = r"Improvement Suggestions:\n([\s\S]*?)(?=\nJob Role Match:|$)"
        jobRoleRegex = r"Job Role Match:\n([\s\S]*)"

        # General advice
        m = re.search(generalRegex, content, re.I)
        if m and m.group(1):
            generalAdvice = m.group(1).strip()

        # Suggestions
        m = re.search(suggestionsRegex, content, re.I)
        if m and m.group(1):
            lines = m.group(1).split('\n')
            for line in lines:
                line = line.replace('**', '').replace('-', '').strip()
                if line and not re.match(r"Job Role Match:", line, re.I):
                    improvementSuggestions.append(line)
        while len(improvementSuggestions) < 4:
            improvementSuggestions.append("No suggestion available.")

        # Job role match
        m = re.search(jobRoleRegex, content, re.I)
        if m and m.group(1):
            lines = [l.strip() for l in m.group(1).split('\n') if l.strip()]
            jobRoleMatch = lines[0] if lines else "No suitable job role identified."
        else:
            jobRoleMatch = "No suitable job role identified."

        return {'generalAdvice': generalAdvice,
                'suggestions': improvementSuggestions,
                'jobRoleMatch': jobRoleMatch}

    parsedData = parseResumeAnalysis(suggestions)

    # --- Calibrate score based on AI advice ---
    display_score = calibrated_score(overall_score, parsedData['generalAdvice'])

    return jsonify({'score': display_score, 'suggestions': suggestions})

@app.route('/get-job-categories', methods=['GET'])
def get_job_categories():
    try:
        categories = df['jobtitle'].dropna().unique().tolist()
        categories.sort()
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
