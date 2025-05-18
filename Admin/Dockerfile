# Use official Python 3.11 image
FROM python:3.11-slim

# Expose the Streamlit port
EXPOSE 8083

# Set working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Use environment variables securely from a .env file (recommended in production via Docker CLI)
# Entry point for the Streamlit app
ENTRYPOINT [ "streamlit", "run", "admin.py", "--server.port=8083", "--server.address=0.0.0.0" ]
