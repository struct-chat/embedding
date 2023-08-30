# Use Python 3.8 as base image
FROM python:3.8

# Set working directory inside the container
WORKDIR /app

# Download the model during build
RUN pip install --no-cache-dir sentence_transformers transformers nltk
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('intfloat/e5-small-v2')"
RUN python -c "import nltk; nltk.download('punkt')"

# Install required Python packages
RUN pip install --no-cache-dir Flask gunicorn numpy

# Copy the Flask app to the container
COPY app.py .

# Expose the port the app runs on
EXPOSE 10002

# Run the app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:10002", "app:app"]
# CMD ["python", "-u", "app.py"]
