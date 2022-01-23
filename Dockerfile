FROM python:3.9

RUN pip install --upgrade pip

# Create working directory
WORKDIR /app

# Copy requirements. From source, to destination.
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port
EXPOSE 8080

# copying all files over. From source, to destination.
COPY . /app

#Run app
CMD streamlit run --server.port 8080 --server.enableCORS false app.py