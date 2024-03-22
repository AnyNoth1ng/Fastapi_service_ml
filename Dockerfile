FROM python:3.11-slim
COPY ./requirements.txt /service/requirements.txt
COPY . /service
WORKDIR /service
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
# CMD ["uvicorn", "app.main:app"]
