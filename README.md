
### CV Parser & Shortlister (Healthcare)

Flask app that ingests CVs (PDF/DOCX), extracts key signals (contact, skills, education, experience, location), and ranks candidates against a job description on a 1–100 scale. The UI shows ranked candidate cards with a concise bullet summary and supports CSV export.
## Features

- Lightweight simple web app
- Handle multiples documents types (pdf, doc, docx)
- Extract relevant informations (candidate's name, email id, phone number, skills)
- Rank candidates based on similarity scores between resumes and job description.


## Quick Start

1) Clone & enter
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

2) Create & activate a virtual environment

Windows (PowerShell)

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip


macOS / Linux

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

3) Install dependencies
pip install -r requirements.txt

4) Run the app

Windows

set FLASK_APP=app.py && python -m flask run


macOS / Linux

FLASK_APP=app.py python -m flask run


 ## Open http://127.0.0.1:8080/

 ## How to Use

Upload CVs (PDF/DOCX) on the home page.

Paste a Job Description and submit.

Open View results to see ranked candidate cards:

Score (1–100), Name, Email, Phone

Location, Experience (yrs), Education, Skills

Concise bullet Summary

View CV button per candidate

# Project Structure
Project Structure
=================
.
├─ app.py
├─ requirements.txt
│
├─ Data/
│  └─ skill_red.csv
│
├─ files/
│  ├─ resumes/           # local uploads
│  └─ outputs/           # generated data
│     ├─ Candidates.csv
│     ├─ latest_shortlist.json
│     └─ latest_jd.json
│
├─ templates/
│  ├─ index.html
│  └─ results.html
│
├─ extract_txt.py
├─ extract_entities.py
├─ txt_processing.py
├─ txt_to_features.py
├─ model.py
└─ summarizer.py


Download CSV if needed.
## Screenshots

<img width="1900" height="1053" alt="image" src="https://github.com/user-attachments/assets/f56a0db3-f12d-44ce-af99-afb0fe67f58b" />

##### Extracted informations
<img width="1897" height="1013" alt="image" src="https://github.com/user-attachments/assets/a555af81-eb9a-439d-a4f2-297a137f93d7" />



