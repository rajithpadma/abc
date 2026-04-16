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
- `VISION_MODEL_PATH` (defaults to `src/vision/model` if empty)
- `EXPORT_PATH` (default: `exports`)

## 2) Build/start commands

This repo already includes:

- `render.yaml`
- `Procfile`
- `wsgi.py`

Render can use either:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn wsgi:application --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`

## 3) Vision model setup (.h5)

For image analysis to work, `.h5` model files must exist in:

- `src/vision/model/`

Expected naming:

- `<Product Name>_good_bad_classifier.h5`

Example:

- `AirChef Fryo_good_bad_classifier.h5`

If models are missing, API returns a graceful error payload instead of failing server requests.

## 4) Verify after deploy

- Health: `GET /api/health`
- Login: `POST /api/auth/login`
- Chat: `POST /api/chat`
- Image: `POST /api/analyze-image` with form-data image file

## 5) Common production issues

- **MongoDB not connecting**: whitelist Render outbound IPs in Atlas or allow access from anywhere (`0.0.0.0/0`) temporarily.
- **Chat API fallback mode**: missing/invalid `OPENROUTER_API_KEY`.
- **Image API unavailable**: TensorFlow/model files missing or incompatible `.h5` model.
