# 🛡️ JobGuard — Explainable Fake Job Detection

> Paste a job posting, get a scam verdict **and the reasons behind it**. A FastAPI backend serves the ML model, SHAP explanations, rule-based red flags and an AI risk summary; a Streamlit dashboard sits on top.

[![CI](https://github.com/patilrakshita2203-prog/JobGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/patilrakshita2203-prog/JobGuard/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red?logo=streamlit)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-explainable_AI-green)](https://shap.readthedocs.io)

**Live demo:** _add your Streamlit link here after deploying (see [Deployment](#deployment))_

---

## What it does

Fake job postings target freshers with registration fees, "guaranteed" placements and WhatsApp-only hiring. JobGuard scores a posting and explains *why*:

1. **ML score** — TF-IDF + Logistic Regression gives a fraud probability and a 0-100 trust score.
2. **SHAP explanation** — the words that pushed the score towards fake or genuine.
3. **Red-flag rules** — fees, WhatsApp/Telegram hiring, free-email recruiters, guaranteed jobs, sensitive-data requests (English + Hindi keywords).
4. **AI risk analysis** — Gemini summarises the verified facts; without an API key a deterministic rule-based summary is used instead.

> The ML model alone can miss India-style scams (see [Limitations](#limitations)), which is why the rule-based red flags run next to it.

## Features

| Feature | Description |
|---|---|
| 🔐 Login | Sign in with Google or with an email and password (accounts, hashed passwords, "keep me signed in") |
| 🔍 Single analyzer | Paste a posting → verdict, trust meter, SHAP words, red flags, AI summary, PDF report |
| 📁 Bulk CSV | Score many postings at once through the API's batch endpoint, download results |
| 🌐 Live scanner | Scrape Naukri / Internshala with Selenium and score each job (local use; needs Chrome) |
| 💰 Salary checker | Flags salaries far above the benchmark range for a role |
| 🧠 Skill check | Compares the skills in a posting with a small built-in reference map for the detected role (curated examples, not live data) |
| 🏢 Company check | Looks for a website whose domain matches the company name (optional, slower) |
| 📊 Analytics | Real model metrics, strongest scam-indicating terms, recent prediction history |
| 🔌 REST API | `/predict`, `/predict/batch`, `/analyze`, `/stats`, `/history`, `/health` with OpenAPI docs |

## Architecture

```
┌──────────────────────┐   HTTP (JSON / CSV)   ┌─────────────────────────────────────────┐
│  Streamlit dashboard │ ────────────────────▶ │  FastAPI backend (main.py)              │
│  app.py              │ ◀──────────────────── │  /predict  /predict/batch  /analyze     │
│  api_client.py       │                       │  /stats  /history  /health              │
└──────────┬───────────┘                       └───────────────┬─────────────────────────┘
           │ API unreachable?                                   │
           ▼                                                    ▼
  falls back to the local model          TF-IDF + Logistic Regression ─▶ SHAP explainer
  (JOBGUARD_MODE=auto)                   Red-flag rules ─▶ Rule engine ─▶ Gemini / rule-based summary
```

The dashboard calls the API when it is reachable and otherwise loads the model itself, so it also works as a single deployed app.

## Project structure

```
JobGuard/
├── app.py                      # Streamlit dashboard (entry point)
├── main.py                     # FastAPI backend (entry point)
├── src/
│   ├── config.py               # all settings (env vars / Streamlit secrets)
│   ├── api_client.py           # dashboard -> API client
│   ├── auth/                   # login page, user store (password hashing, sessions)
│   ├── preprocessing/          # data_cleaning.py (NLTK text cleaning)
│   ├── ml/                     # vectorizer, SMOTE, model_training, hybrid_model_training, predictors
│   ├── explainability/         # shap_explainer, rule_engine, red_flags, company_verifier,
│   │                           # prompt_builder, llm_analyzer, fallback_analyzer, knowledge_graph_realtime
│   ├── reporting/              # pdf_reporter.py
│   └── scraper/                # Naukri + Internshala scrapers, naukri_parsing.py, selenium_scraper.py
├── models/                     # best_model.joblib, tfidf_vectorizer.joblib, metrics JSON
├── sample_data/jobs.csv        # demo CSV for the bulk tab
├── tests/                      # pytest suite (API, dashboard, rules, client, training)
├── screenshots/
├── .github/workflows/ci.yml    # runs the tests on every push
└── requirements.txt            # pinned runtime dependencies
```

## Quick start

```bash
git clone https://github.com/patilrakshita2203-prog/JobGuard.git
cd JobGuard

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # add GEMINI_API_KEY (optional)

# terminal 1 - backend (docs at http://127.0.0.1:8000/docs)
uvicorn main:app --reload

# terminal 2 - dashboard (http://localhost:8501)
streamlit run app.py
```

The first time, click *Create account* on the login page. The sidebar shows whether the dashboard is connected to the API or running in local mode.

To retrain the models: install the training packages listed at the top of `requirements.txt`, then run `python -m src.ml.model_training` (needs `data/fake_job_postings.csv` from the Kaggle dataset below).

## API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "title": "Work From Home",
  "description": "URGENT hiring! Registration fee Rs 500 required. Contact on WhatsApp."
}'
```

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Score one posting: verdict, probability, trust score, SHAP words, red flags |
| `POST /predict/batch` | Upload a CSV with a `description` column, get the CSV back with predictions |
| `POST /analyze` | Rule-engine facts + red flags + AI (or rule-based) risk summary |
| `GET /stats` | Training metrics for all models |
| `GET /history` | Recent predictions logged by the API |
| `GET /health` | Liveness, model status, whether Gemini is configured |

## Configuration

Copy `.env.example` to `.env`. On Streamlit Community Cloud use *Secrets* instead (same names, TOML format).

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | – | Enables Gemini summaries (falls back to rules without it) |
| `JOBGUARD_API_URL` | `http://127.0.0.1:8000` | Where the dashboard finds the API |
| `JOBGUARD_MODE` | `auto` | `auto` use API if reachable, `api` require it, `local` never call it |
| `JOBGUARD_API_KEY` | – | If set, the API requires an `X-API-Key` header (the dashboard sends it) |
| `JOBGUARD_CORS_ORIGINS` | localhost:8501 | Browser origins allowed to call the API |
| `JOBGUARD_RATE_LIMIT` / `_BATCH_RATE_LIMIT` | 60 / 10 per min | Per-IP request limits |
| `JOBGUARD_MAX_UPLOAD_MB` / `_MAX_BATCH_ROWS` | 5 / 2000 | Batch upload limits |
| `JOBGUARD_ENABLE_SCRAPING` | `true` | Set `false` on hosts without Chrome |
| `JOBGUARD_AUTH_REQUIRED` | `true` | Set `false` to skip the login page while developing |
| `JOBGUARD_AUTH_DB_PATH` | `data/users.db` | Where email/password accounts are stored |
| `JOBGUARD_SESSION_DAYS` | `30` | How long "Keep me signed in" lasts |
| `JOBGUARD_ALLOW_SIGNUP` | `true` | Set `false` to stop new accounts being created |

## Dataset

**EMSCAD** (Employment Scam Aegean Dataset) — [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction): 17,880 postings, 866 fraudulent (4.84%). The CSV is not stored in this repo.

## Model performance

Stratified 80/20 split → **3,576 test postings (173 fraudulent)**. SMOTE (`sampling_strategy=0.3`) is applied to the training split only. Source: `models/training_results.json`.

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (deployed)** | 0.785 | 0.884 | **0.832** | 0.988 |
| Random Forest | 0.922 | 0.682 | 0.784 | 0.986 |
| Hybrid TF-IDF + BERT + LR (experiment) | 0.655 | 0.890 | 0.755 | 0.983 |
| XGBoost | 0.048 | 1.000 | 0.092 | 0.458 |

Logistic Regression's confusion matrix: 3,361 genuine correctly kept, 42 false alarms, 153 scams caught, 20 scams missed.

**About the XGBoost row:** those numbers come from a training bug that is now fixed in `src/ml/model_training.py` (XGBoost was trained on dense arrays but scored on sparse ones, so it labelled every posting fake). The table above is from the saved run; re-run `python model_training.py` to refresh it. Logistic Regression is expected to stay the best model.

## Limitations

- **Dataset age and language.** EMSCAD is an older, mostly English, US-centred dataset. Typical Indian fee-and-WhatsApp scams look different, so the ML score can be low for them. That is why red-flag rules run alongside the model.
- **Few positives.** 866 fraudulent postings; metrics on 173 test positives are noisy.
- **The saved models were trained before two fixes.** TF-IDF was fitted on the full dataset before the train/test split (a small leak), and the best model is picked by test-set F1. `src/ml/model_training.py` now fits TF-IDF on the training split only; retrain to apply it. For a rigorous comparison, add a separate validation split.
- **The model learned some wrong word associations.** For example "whatsapp" and "registration" lean *genuine* in the old US data. SHAP now explains the model faithfully, which exposes this. The rule-based red flags exist to cover it.
- **No duplicate audit.** The dataset contains near-duplicate postings that may appear in both train and test splits and flatter the scores.
- **Not a verdict.** JobGuard is an educational tool. Verify employers through official channels and report fraud at [cybercrime.gov.in](https://cybercrime.gov.in).
- **Scraping** may break when portals change their pages and can conflict with their terms of use; it is off by default on hosted deployments.

## Testing

```bash
pip install -r requirements.txt pytest httpx
pytest -q
```

Tests run offline (no Gemini key, web search or Chrome needed). GitHub Actions runs them on every push.

## Login

Everything is behind a login page. People can sign in two ways:

- **Email and password.** Accounts live in a local SQLite file (`data/users.db`). Passwords are never stored, only a salted scrypt hash. Five wrong passwords lock an account for 15 minutes. "Keep me signed in" uses a random session token in a browser cookie, so a refresh does not log you out; only a hash of the token is kept on the server.
- **Google.** Uses Streamlit's built-in sign-in. It needs a free Google OAuth client:
  1. In the [Google Cloud Console](https://console.cloud.google.com/) create a project, set up the *OAuth consent screen* (External, add yourself as a test user), then *Credentials → Create credentials → OAuth client ID → Web application*.
  2. Add the redirect URI `http://localhost:8501/oauth2callback` (and, after deploying, `https://<your-app>.streamlit.app/oauth2callback`).
  3. Create `.streamlit/secrets.toml` (it is git-ignored) with:
     ```toml
     [auth]
     redirect_uri = "http://localhost:8501/oauth2callback"
     cookie_secret = "any-long-random-string"
     client_id = "xxxx.apps.googleusercontent.com"
     client_secret = "xxxx"
     server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
     ```
     A random `cookie_secret` comes from `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
  4. Restart the dashboard. The "Continue with Google" button turns on.

Good to know: there is no "forgot password" email yet, and on Streamlit Community Cloud the accounts file is wiped whenever the app restarts (use Google sign-in there, or point `JOBGUARD_AUTH_DB_PATH` at a persistent disk). Passwords and session cookies should only be used over HTTPS in production.

## Deployment

**Dashboard on Streamlit Community Cloud (free):**
1. <https://share.streamlit.io> → *New app* → this repo, branch `main`, main file `app.py`.
2. *Advanced settings → Secrets*:
   ```toml
   GEMINI_API_KEY = "your_key"
   JOBGUARD_ENABLE_SCRAPING = "false"

   [auth]
   redirect_uri = "https://<your-app>.streamlit.app/oauth2callback"
   cookie_secret = "any-long-random-string"
   client_id = "xxxx.apps.googleusercontent.com"
   client_secret = "xxxx"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
3. With no API configured the dashboard runs in local mode (the sidebar says so); only live scraping is unavailable.

**API on Render (optional, so the dashboard really talks to the API):**
1. <https://render.com> → *New → Web Service* → this repo.
2. Build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, health check path `/health`.
3. Environment: `GEMINI_API_KEY`, `JOBGUARD_API_KEY` (any long random string), `JOBGUARD_CORS_ORIGINS` (your Streamlit URL), `JOBGUARD_LOG_PREDICTIONS=false`.
4. Add to the Streamlit secrets: `JOBGUARD_API_URL = "https://<your-service>.onrender.com"` and the same `JOBGUARD_API_KEY`, then reboot the app. The sidebar should say *FastAPI backend connected*.

The free Render tier sleeps when idle (first request ~30-60 s; the dashboard falls back to its local model meanwhile) and has little RAM. Keep `JOBGUARD_API_KEY` set and `JOBGUARD_CORS_ORIGINS` limited to your dashboard URL.

## Roadmap

- Duplicate-aware re-split and retraining on Indian job postings
- Fix and re-evaluate the XGBoost run
- Recruiter reputation scoring and email-domain checks beyond free-mail detection
- Multi-portal scraping and an ATS resume-compatibility checker
