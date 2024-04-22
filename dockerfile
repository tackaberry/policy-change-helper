FROM python:3.9
EXPOSE 8080
WORKDIR /app
COPY App.py requirements.txt .env ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "App.py", "--server.port=8080", "--server.address=0.0.0.0"]