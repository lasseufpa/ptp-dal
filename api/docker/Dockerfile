FROM python:3.6

LABEL maintainer="Camila Novaes <novaes.csw@gmail.com>"

# Install requirements
COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Add source files
COPY ptp/compression.py /app/ptp/
COPY api /app/api/

# Define environment variables
ENV DS_PATH /app/api/datasets/
ENV PYTHONDONTWRITEBYTECODE 1

WORKDIR /app/api
CMD ["gunicorn", "-b", "0.0.0.0:80", "run:app", "--timeout", "120", \
"--log-level", "DEBUG", "--access-logfile", "-", "--error-logfile", "-"]

