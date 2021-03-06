# write some code to build your image
FROM python:3.8.13-buster

COPY requirements.txt /requirements.txt
COPY api /api
COPY equations /equations
COPY math_model_v1 /math_model_v1


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
