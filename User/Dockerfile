# Use official Python image
FROM python:3.11

# Set working directory
WORKDIR /app

# Expose port for Streamlit app
EXPOSE 8084

# Copy dependencies and install them
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Run the Streamlit app on port 8084
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8084", "--server.address=0.0.0.0"]
