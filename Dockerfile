FROM python:3.12-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY parts/ parts/
COPY benchmarks/ benchmarks/
COPY verification/ verification/
COPY run.sh .
RUN mkdir -p results
CMD ["bash", "run.sh"]
