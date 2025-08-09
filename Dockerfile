FROM python:3.10
WORKDIR /app

# Install deps
COPY requirements.txt .
# make sure requirements.txt includes: flask, gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . .

# Expose the web port (optional but nice)
EXPOSE 8000

# Run Flask with gunicorn (Green Unicorn)
# format: gunicorn <module>:<app>  -> api.app:app
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "120", "-b", "0.0.0.0:8000", "api.app:app"]
