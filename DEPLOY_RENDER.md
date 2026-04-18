# Render Deployment Guide

## 1) Required environment variables in Render

Set these in **Render > Service > Environment**:

- `MONGODB_URI` = your MongoDB Atlas URI
- `DATABASE_NAME` = `Product_Database` (or your DB name)
- `OPENROUTER_API_KEY` = your OpenRouter key
- `FLASK_SECRET_KEY` = any long random string

Optional:

- `OPENROUTER_MODEL`
- `MODEL_PRIORITY`
- `EXPORT_PATH` (default: `exports`)

## 2) Build/start commands

This repo already includes:

- `render.yaml`
- `Procfile`
- `wsgi.py`

Render can use either:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn wsgi:application --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

This default Render setup is optimized for `512 MB` instances. Chat and login work without TensorFlow.

## 3) Vision model setup (.h5)

Image analysis is optional in production. The base deploy now skips TensorFlow to keep Render memory usage low.

If you want image analysis to work, deploy with both of these:

1. Install extra dependencies:
   - `pip install -r requirements.txt -r requirements-vision.txt`
2. Place `.h5` model files in:
   - `src/vision/model/`

Expected naming:

- `<Product Name>_good_bad_classifier.h5`

Example:

- `AirChef Fryo_good_bad_classifier.h5`

If models are missing, API returns a graceful message and the chat UI still works.

## 4) Verify after deploy

- Health: `GET /api/health`
- Login: `POST /api/auth/login`
- Chat: `POST /api/chat`
- Image: `POST /api/analyze-image` with form-data image file

## 5) Common production issues

- **MongoDB not connecting**: whitelist Render outbound IPs in Atlas or allow access from anywhere (`0.0.0.0/0`) temporarily.
- **Login works locally but not on Render**: the app now falls back to the bundled CSV data automatically when MongoDB is unavailable.
- **Chat API fallback mode**: missing/invalid `OPENROUTER_API_KEY`. The app still replies with the built-in fallback assistant.
- **Image API unavailable**: TensorFlow extra dependencies or `.h5` model files are missing. Chat remains available.
