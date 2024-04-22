# import libraries
import streamlit as st

import json
from dotenv import dotenv_values
import vertexai

from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.discoveryengine_v1.types import common, search_service
from StreamlitGauth.google_auth import Google_auth

from dotenv import dotenv_values
from PyPDF2 import PdfReader 
import time

import pandas as pd
import os

import urllib.parse


session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
st_base_url = urllib.parse.urlunparse([session.client.request.protocol, session.client.request.host, "", "", "", ""])

PROJECT = os.environ.get('PROJECT')
LOCATION = os.environ.get("LOCATION")
DATASTORE = os.environ.get("DATASTORE")
BUCKET = os.environ.get("BUCKET")

CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

# Define the scopes required for your app
scopes = ['openid', 'email', 'profile']

redirect_uri = os.environ.get("REDIRECT_URI")
redirect_uri = redirect_uri or st_base_url

if 'login' not in st.session_state or "authenticated" not in st.session_state.login:
    st.session_state.login = Google_auth(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET, redirect_uri=redirect_uri)

    if st.session_state.login is not None and "authenticated" in st.session_state.login:
        # your streamlit applciation
        pass

    else:
        st.stop()
           
vertexai.init(project=PROJECT, location=LOCATION)

MODEL_NAME = "gemini-1.5-pro-preview-0215"
TOTAL_TOKENS = 1000000
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 8192
TOP_P = 0.8
TOP_K = 40

SUMMARIZE_PROMPT = """You are an intelligent policy analyst helping on determine what are the NEEDED CHANGES to be made to an EXISTING POLICY in order to implement the PROPOSED CHANGE on an an EXISTING POLICY.
Please summarize the relevant portions of the EXISTING POLICY when explaining the NEEDED CHANGES.
Strictly Use ONLY the following pieces of context to determine the NEEDED CHANGES for the PROPOSED CHANGE.
Do not respond with  a summary if it is not relevant to the PROPOSED CHANGE.

EXISTING POLICY: 
{policy}

PROPOSED CHANGE:
{proposal}

NEEDED CHANGES:
"""

EXTRACT_PROMPT = """You are an intelligent policy analyst helping on determine what are the NEEDED CHANGES to be made to an EXISTING POLICY in order to implement the PROPOSED CHANGE on an an EXISTING POLICY.
Please summarize the relevant portions of the EXISTING POLICY when explaining the NEEDED CHANGES.
Strictly Use ONLY the following pieces of context to determine the NEEDED CHANGES for the PROPOSED CHANGE.
Do not respond with  a summary if it is not relevant to the PROPOSED CHANGE.

EXISTING POLICY: 
{policy}

PROPOSED CHANGE:
{proposal}

NEEDED CHANGES:
"""

def run_prompt_preview(prompt):
    vertexai.init(project=PROJECT, location=LOCATION)
    model = GenerativeModel(MODEL_NAME)
    responses = model.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": st.session_state.max_output_tokens,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p
        },
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=True,
    )
    full_response = ""
    for response in responses:
        full_response += response.text

    return full_response

def run_prompt(prompt):
    # use preview approach for preview models
    if (MODEL_NAME in ["gemini-1.5-pro-preview-0215"]):
        return run_prompt_preview(prompt)
    # run prompt using vertex ai
    parameters = {
        "candidate_count": 1,
        "temperature": st.session_state.temperature,
        "max_output_tokens": st.session_state.max_output_tokens,
        "top_p": st.session_state.top_p,
        "top_k": st.session_state.top_k,
    }

    model = TextGenerationModel.from_pretrained(MODEL_NAME)
    response = model.predict(
        prompt,
        **parameters
    )
    return response.text

def extract_text(policy, proposal):
    extract_prompt = st.session_state.extract_prompt
    prompt = extract_prompt.format(policy=policy, proposal=proposal)
    return run_prompt(prompt)

def summarize_policy(policy, proposal):
    summarize_prompt = st.session_state.summarize_prompt
    prompt = summarize_prompt.format(policy=policy, proposal=proposal)
    return run_prompt(prompt)

def get_file_content(link):
    # download file from google cloud storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob(link.replace(f"gs://{BUCKET}",""))
    blob.download_to_filename("temp.pdf")
    reader = PdfReader('temp.pdf') 

    chunks = []
    range = []
    p = 1
    chunk = ""
    token_size = (TOTAL_TOKENS - st.session_state.max_output_tokens) * 4
    for page in reader.pages:

        # if length of chunk is greater than 15000 characters, then add it to the list
        if len(chunk) > token_size:
            chunks.append([chunk,range])
            range = []
            chunk = ""

        chunk += page.extract_text()
        range.append(p)
        p += 1


    chunks.append(chunk)

    return chunks



def search_sample(
    project_id,
    location,
    data_store_id,
    search_query,
):
    #  For more information, refer to:
    # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if location != "global"
        else None
    )

    # Create a client
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    # The full resource name of the search engine serving config
    # e.g. projects/{project_id}/locations/{location}/dataStores/{data_store_id}/servingConfigs/{serving_config_id}
    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        serving_config="default_config",
    )

    # Optional: Configuration options for search
    # Refer to the `ContentSearchSpec` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
    content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
        # For information about snippets, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/snippets
        snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
            return_snippet=True,
        ),
    )

    # Refer to the `SearchRequest` reference for all supported fields:
    # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=5,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    response = client.search(request)

    return response

def response_to_df(response):

    df = pd.DataFrame(columns=['Id', 'Page', 'Title','Link','Snippet'])

    pages = tuple(search_service.SearchResponse.to_json(x) for x in response.pages)

    firstPage = json.loads(pages[0])
    meta = {
        "totalSize": firstPage["totalSize"],
    }


    p=0
    for page in pages:
        page = json.loads(page)
        p += 1

        for result in page["results"]:
            # name = result.document.name
            id = result["document"]["id"]
            title = result["document"]["derivedStructData"]["title"]
            link = result["document"]["derivedStructData"]["link"].replace(f"gs://{BUCKET}/","")
            snippet = result["document"]["derivedStructData"]["snippets"][0]["snippet"]
            df.loc[len(df)] = {'Id': id, 'Page': p, 'Title':title, 'Link':link, 'Snippet':snippet}

    return [df,meta]

if "showOne" not in st.session_state:
    st.session_state.showOne = False
if "showTwo" not in st.session_state:
    st.session_state.showTwo = False


st.title('Policy Helper')
question=st.text_input("Outline your search or policy change")

if question:
    if st.session_state.showTwo:
        st.session_state.showOne = False
    else:
        st.session_state.showOne = True


def analyze_this(link):
    st.session_state.link = link
    st.session_state.showOne = False
    st.session_state.showTwo = True

def back_to_results():
    st.session_state.showOne = True
    st.session_state.showTwo = False

with st.sidebar:
    st.markdown("# Settings")
    
    summarize_prompt = st.text_area("Summarize Prompt", value=SUMMARIZE_PROMPT)
    st.session_state.summarize_prompt = summarize_prompt

    extract_prompt = st.text_area("Extract Prompt", value=EXTRACT_PROMPT)
    st.session_state.extract_prompt = extract_prompt

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=TEMPERATURE,
    )
    st.session_state.temperature = temperature

    top_p = st.slider('Top P:', min_value=0.0, max_value=1.0, value=TOP_P)
    st.session_state.top_p = top_p

    top_k = st.slider('Top K:', min_value=0, max_value=40, value=TOP_K)
    st.session_state.top_k = top_k

    max_output_tokens = st.slider(
        "Max Output Tokens:", min_value=0, max_value=1000, value=MAX_OUTPUT_TOKENS
    )
    st.session_state.max_output_tokens = max_output_tokens


if st.session_state.showOne:

    response2 = search_sample(PROJECT, "global", DATASTORE, question)
    [df, meta] = response_to_df(response2)

    with st.expander("See results", expanded=False):
        st.dataframe(df)

    for row in df.itertuples():
        st.markdown(f"# {row.Title}")
        st.markdown(row.Snippet, unsafe_allow_html=True)
        st.button("Analyze this", key=row.Id, on_click=analyze_this, args=(row.Link,))
        st.divider()
        
if st.session_state.showTwo:

    st.button( "Back", on_click=back_to_results )

    chunks = []
    with st.spinner("Loading document..."):
        st.markdown(f"# {st.session_state.link}")

        chunks = get_file_content(st.session_state.link)

        st.write(f"Doc: {st.session_state.link}")
        st.write(f"Chunks: {len(chunks)}")

        SLEEP_TIMEOUT = 5
    
    paragraphs = []

    breakdown, summary = st.tabs(["Page Breakdown", "Summary"])

    if len(chunks) > 1:
        chunk_count = 1
        with breakdown:
            for chunk in chunks:
                with st.spinner(f"Analyzing chunk #{chunk_count}"):
                    chunk_count += 1
                    text = chunk[0]
                    paragraph = extract_text(text, question)
                    paragraphs.append(paragraph)
                    st.write( "Pages: {first} - {last}".format(first=str(chunk[1][0]), last=str(chunk[1][-1])) )
                    st.write(paragraph)
                    st.divider()
                time.sleep(SLEEP_TIMEOUT)
    else:
        paragraphs.append(chunks[0])
        with breakdown:
            with st.spinner("Analyzing..."):
                st.write("Summary is ready")
    
    with summary:
        with st.spinner("Analyzing..."):
            response = summarize_policy("\n\n".join(paragraphs), question)
            st.write(response)
