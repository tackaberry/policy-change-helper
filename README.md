
# Document Compare Helper

The Document Compare Helper is a tool to help users understand the differences between two documents


## Setup
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Set `.env` file
```bash
PROJECT = 'ppppppp'
LOCATION = 'llllllll'
DATASTORE = 'dddddddd'
BUCKET = 'bbbbbbbb'
```

## Auth Setup
```bash
gcloud auth application-default login
```

## Run
```bash
streamlit run App.py
```

## Docker
Set the env var `GOOGLE_APPLICATION_CREDENTIALS` to the credentials file created on `gcloud auth application-default login` (usually somewhere in ~/.config/gcloud/...)

```bash
# Build
docker build -t cloudrun-policy-change-helper:latest .

# Run
docker run --rm -it -p 8080:8080 \
-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/google_auth.json \
-v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/keys/google_auth.json:ro \
cloudrun-policy-change-helper:latest
```

Visit localhost:8080