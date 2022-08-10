FROM python:3.8
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
WORKDIR /app
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "ShipClassifierAPI:app", "--host", "0.0.0.0"] 