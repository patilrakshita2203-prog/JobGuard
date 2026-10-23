# Deploying JobGuard

JobGuard has two parts: the **FastAPI backend** (`main.py`) and the **Streamlit dashboard** (`app.py`).
The dashboard works on its own (it loads the model itself), so the quickest deployment is the dashboard alone.

---

## 0. Before you push to GitHub

1. **Rotate your Gemini key** if the old `.env` was ever shared, and keep `.env` out of git (it is in `.gitignore`).
2. Make sure the two model files are committed (the updated `.gitignore` allows them):
   ```bash
   git add models/best_model.joblib models/tfidf_vectorizer.joblib sample_data/jobs.csv
   git add .
   git status            # check: no .env, no .db, no big CSVs
   git commit -m "Add FastAPI-connected dashboard, tests, CI and deployment files"
   git push
   ```
3. Run the tests locally: `pip install -r requirements-dev.txt && pytest -q`.

## 1. Dashboard on Streamlit Community Cloud (free, easiest)

1. Go to <https://share.streamlit.io> → **New app** → pick this repo, branch `main`, main file `app.py`.
2. **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_key"
   JOBGUARD_ENABLE_SCRAPING = "false"
   ```
3. Deploy. With no API configured the app runs in *local mode* (the sidebar says so) — everything works except live scraping.
4. Copy the public URL into the README's **Live demo** line.

## 2. Add the FastAPI backend on Render (so the dashboard really talks to the API)

1. Go to <https://render.com> → **New → Blueprint** → select this repo (it reads `render.yaml`).
2. In the service's **Environment** tab set `GEMINI_API_KEY`, and set `JOBGUARD_CORS_ORIGINS` to your Streamlit URL.
   Copy the generated `JOBGUARD_API_KEY`.
3. After the deploy, open `https://<your-service>.onrender.com/docs` to confirm the API is up.
4. In Streamlit **Secrets** add:
   ```toml
   JOBGUARD_API_URL = "https://<your-service>.onrender.com"
   JOBGUARD_API_KEY = "the key Render generated"
   ```
   Reboot the Streamlit app. The sidebar should now say **FastAPI backend connected**.

Notes
- Render's free tier sleeps after inactivity, so the first request can take ~30-60 s. The dashboard falls back to the local model in `auto` mode if the API does not answer in time.
- The free tier has little RAM (512 MB). If the service is killed for memory, move to a paid instance or use the dashboard-only deployment.
- The free tier's disk is temporary, so prediction history is switched off there (`JOBGUARD_LOG_PREDICTIONS=false`).

## 3. Docker (any VPS or your laptop)

```bash
cp .env.example .env        # fill GEMINI_API_KEY (optional)
docker compose up --build
```

- Dashboard: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>

The compose file keeps the prediction history in a named volume and turns live scraping off (no Chrome in the image).

## 4. Production checklist

- [ ] `JOBGUARD_API_KEY` set on the API and in the dashboard secrets
- [ ] `JOBGUARD_CORS_ORIGINS` lists only your dashboard URL (not `*`)
- [ ] HTTPS in front of the API (Render and Streamlit Cloud provide it)
- [ ] `JOBGUARD_ENABLE_SCRAPING=false` on hosted deployments
- [ ] Gemini key stored only in the host's secret manager
- [ ] README shows the live link and fresh screenshots of the new dashboard
