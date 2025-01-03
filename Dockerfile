FROM python:3.10.12

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY . ./

ENTRYPOINT ["python3", "-u", "inference_remote.py"]
