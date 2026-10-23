# What changed in this upgrade

Copy these files over your repo (same paths). Nothing else in your project was touched:
training scripts, scrapers, `shap_explainer.py`, `pdf_reporter.py`, `models/`, `screenshots/` are unchanged.

## Modified existing files (10)
| File | Change |
|---|---|
| `app.py` | Connected to the FastAPI backend (auto fallback to local model); real F1 in sidebar; honest header/footer text; demo buttons now fill the form; results survive button clicks (PDF / AI analysis / download); bulk and live-scanner code moved inside their tabs; AI analysis uses the *selected* job; red-flag + AI sections; fake keyword chart replaced by real model coefficients; recent-predictions table; GitHub placeholders fixed; no more random "demo" verdicts unless `JOBGUARD_DEMO_MODE=true` |
| `main.py` | Same endpoints/schemas, plus `/analyze`, `/history`; API-key auth, rate limit, upload limits, CORS from env, lifespan startup, absolute model paths, specific exceptions, 503 no longer turned into 500, genuine-side SHAP words fixed (`top_real_words` key), batch now uses `benefits`, handles NaN, vectorised, consistent risk levels |
| `src/explainability/llm_analyzer.py` | Lazy Gemini init, Streamlit-secrets support, timeout, retries, cache, rule-based fallback (no raw "LLM Error"). `generate_ai_analysis(prompt)` still works |
| `src/explainability/company_verifier.py` | Domain must match the company; returns `None` (unknown) on errors instead of `False`; cached |
| `src/explainability/rule_engine.py` | None-safe; `verify_company` flag; adds `red_flags` |
| `src/explainability/prompt_builder.py` | Tells the LLM when the website check was impossible; includes red flags |
| `data_cleaning.py` | Downloads NLTK data only if missing (no network call on every start) |
| `requirements.txt` | UTF-8, runtime only, all missing packages added, pinned (installs from scratch) |
| `.gitignore` | Keeps old rules; allows the 2 model files + `sample_data/`; ignores secrets/db |
| `README.md` | UTF-8; real metrics; real structure; limitations; API, config, deploy |

## New files
`api_client.py`, `jobguard_config.py`, `src/explainability/{red_flags,fallback_analyzer,job_posting}.py`,
`tests/` (48 tests), `requirements-dev.txt`, `requirements-train.txt`, `.env.example`,
`.streamlit/{config.toml,secrets.toml.example}`, `Dockerfile`, `.dockerignore`, `docker-compose.yml`,
`render.yaml`, `.github/workflows/ci.yml`, `docs/DEPLOYMENT.md`, `LICENSE`.

## Not changed on purpose (decide yourself)
- File layout: kept flat so no import breaks. Duplicate screenshots in the repo root and the second `predictor.py` are left alone.
- Training code: `train_xgboost` in `model_training.py` produced a model that labels everything fake (F1 0.09, AUC 0.46). Not touched.
- `google-generativeai` is deprecated upstream in favour of `google-genai`; works today.
