FROM python:3.10-slim-buster

RUN mkdir /src
WORKDIR /src

COPY requirements.txt /src/
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]