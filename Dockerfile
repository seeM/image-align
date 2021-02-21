FROM python:3.7.9-slim-buster

MAINTAINER Wasim Lorgat <mwlorgat@gmail.com>

ENV PYTHONUNBUFFERED 0

WORKDIR /
# main.py assumes data exists in ./data
VOLUME /data

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["python3", "main.py"]
