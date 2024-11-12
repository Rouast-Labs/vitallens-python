FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip setuptools wheel
RUN pip install .
RUN pip install matplotlib pandas
ENTRYPOINT ["python", "examples/test.py"]
CMD ["--api_key", "", "--vitals_path", "examples/sample_vitals_1.csv", "--video_path", "examples/sample_video_1.mp4", "--method", "VITALLENS", "--input_str", "True"]
