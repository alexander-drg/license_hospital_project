# Intelligent Matching Algorithms for Candidate-Job Compatibility Analysis Using NLP

> Bachelor's Thesis Project, Politehnica Bucharest, 2026
> Alexandru Draghici

A hybrid AI-powered recruitment platform that matches candidate CVs to job descriptions using a combination of deterministic feature scoring and semantic analysis via a locally hosted Large Language Model (Ollama).

---

## Screenshots

### Homepage,Upload Interface
<img width="1084" height="540" alt="image" src="https://github.com/user-attachments/assets/f38d2926-f6e3-4831-8adf-e6a05ae6f66d" />

### Ranked Results with Component Score Breakdown
*(example for 1 job description with 2 candidates CVs)*

<img width="884" height="445" alt="image" src="https://github.com/user-attachments/assets/bf94c29c-3bea-4a12-9f4f-a6a582333b1d" />

### AI Analysis Results
<img width="896" height="823" alt="image" src="https://github.com/user-attachments/assets/357b1e01-07ec-4dfa-ae11-d7029093b8ce" />

### SQL Database Analyser
*(selected 5 relevant tables out of all of them)*

<img width="884" height="692" alt="image" src="https://github.com/user-attachments/assets/dc88edf8-4304-45d7-b150-392769e9e0db" />
<img width="872" height="604" alt="image" src="https://github.com/user-attachments/assets/cddb4ba1-99d7-4606-b9f9-299604da5a20" />

*(can also display the raw data from the tables themselves)*

---

## Features

- **Multi-format CV parsing** : PDF, DOCX, TXT, and image-based documents (OCR fallback via Tesseract)
- **Hybrid matching algorithm** : combines skill overlap (Jaccard), experience alignment, location compatibility, and semantic cosine similarity
- **Local LLM integration** : all AI inference via Ollama, no data sent to external APIs
- **SQL Database Analyser** : connects to a MySQL recruitment database, extracts schema, detects relationships, and generates AI-assisted plain-language interpretations
- **Configurable scoring weights** : adjust the importance of each matching dimension per role type
- **Flask web interface** : fully browser-based, no command-line required for normal use

---

## How It Works

The compatibility score M is computed as:

```
M = w1*S + w2*E + w3*L + w4*C
```

Where:
- **S** : Skill similarity (Jaccard coefficient + competency weighting)
- **E** : Experience alignment (duration + topical relevance)
- **L** : Location compatibility (3-tier geographic matching)
- **C** : Semantic cosine similarity (Ollama LLM embeddings)

Default weights: `w1=0.40, w2=0.30, w3=0.10, w4=0.20`

---

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) with at least one model pulled:
  ```bash
  ollama pull llama3
  ```
- MySQL Server (required only for the SQL Analyser feature)
- Tesseract OCR -- [install here](https://github.com/tesseract-ocr/tesseract)

### Installation

```bash
git clone https://github.com/alexander-drg/license_hospital_project.git
cd license_hospital_project
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Running the App

Start Ollama in a separate terminal:
```bash
ollama serve
```

Then launch the Flask app:
```bash
cd app
flask run
```

Open your browser at `http://localhost:5000`

---

## SQL Database Analyser Setup

The SQL Analyser connects to an existing MySQL database. To configure the connection:

1. Make sure MySQL Server is running on your machine or is accessible on your network
2. In the app, navigate to the **SQL Analyser** section from the top menu
3. Enter your connection parameters directly in the interface:
   - **Host** -- e.g. `localhost` or your server IP
   - **Port** -- default MySQL port is `3306`
   - **Database name** -- the name of your recruitment database
   - **Username** -- your MySQL username
   - **Password** -- your MySQL password
4. Click **Connect** -- the app will query the database and display all available tables
5. Select the tables you want to analyse and click **Analyse Selected Tables**

> **Note:** Connection parameters are entered at runtime through the browser interface and are never stored to disk.

---

## Usage

1. **Upload CVs** -- drag and drop or select PDF/DOCX/TXT/image files
2. **Upload a Job Description** -- same supported formats
3. **Run Match** -- click "Match Candidates" to get the ranked list with component scores
4. **AI Analysis** -- click "Generate AI Analysis" for Ollama-generated candidate summaries
5. **SQL Analyser** -- connect to your MySQL database, select tables, and get AI-assisted schema interpretation

---

## Project Structure

```
license_hospital_project/
├── app/
│   ├── app.py                   # Flask routes and orchestration
│   ├── cv.py                    # CV parsing pipeline
│   ├── job.py                   # Job description parsing
│   ├── ranker.py                # Hybrid matching and ranking
│   ├── competency.py            # Competency matching and skill clustering
│   ├── embeddings.py            # Ollama embedding integration
│   ├── ollama_ai.py             # LLM interpretation layer
│   ├── sql_candidate_parser.py  # SQL Analyser module
│   └── models.py                # Data models
├── web/templates/               # Jinja2 HTML templates
│   ├── base.html
│   ├── home.html
│   ├── results.html
│   ├── candidate.html
│   ├── job.html
│   └── sql_analyzer.html
├── data/                        # Persistent JSON storage for parsed profiles
├── requirements.txt
└── README.md
```

---

## Configuration

Scoring weights can be adjusted in `app/config.json`:

```json
{
  "weights": {
    "skills":     0.40,
    "experience": 0.30,
    "location":   0.10,
    "semantic":   0.20
  },
  "ollama_model":       "llama3",
  "ollama_timeout_emb": 30,
  "ollama_timeout_gen": 45
}
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | Python 3, Flask |
| Frontend | HTML, CSS, Jinja2 |
| PDF Extraction | pdfplumber, PyMuPDF |
| OCR | Tesseract via pytesseract |
| NLP / Embeddings | sentence-transformers, Ollama |
| Database | MySQL via mysql-connector-python |
| ML Utilities | scikit-learn, numpy, scipy |

---

## License

This project was developed as a Bachelor's thesis at the National University of Science and Technology POLITEHNICA Bucharest, Faculty of Electronics, Telecommunications and Information Technology.
