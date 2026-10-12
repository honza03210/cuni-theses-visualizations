FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Download the dataset and build every derived table at build time, then drop the raw word
# tables: the running app only needs the theses table and data/derived/ (about 380 MB).
RUN python prepare.py --clean

ENV DASH_DEBUG=0 PORT=8050
EXPOSE 8050
# One worker with threads: each process caches the word tables (about 1.1 GB once all views are used).
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120
