# Use Python 3.10 base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt (or manually install dependencies later)
COPY requirements.txt /app/

# Install the dependencies in the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend project into the container
COPY . /app/

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
