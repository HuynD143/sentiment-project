FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 libfreetype6 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

EXPOSE 8502

CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0", "--server.port=8502"]
