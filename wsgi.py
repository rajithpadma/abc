from main import app, ensure_services_initialized

# Ensure services are initialized when loaded by Gunicorn/Render.
ensure_services_initialized()

application = app
