# import libraries
import streamlit as st
from streamlit import session_state as ss
import json
from dotenv import dotenv_values
import vertexai

from vertexai.language_models import TextGenerationModel

from google.cloud import storage
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.discoveryengine_v1.types import common, search_service

from dotenv import dotenv_values
from PyPDF2 import PdfReader 
import time


import pandas as pd


config = dotenv_values(".env")

PROJECT = config["PROJECT"]
LOCATION = config["LOCATION"]
DATASTORE = config["DATASTORE"]
BUCKET = config["BUCKET"]

vertexai.init(project=PROJECT, location=LOCATION)

MODEL_NAME = "text-bison-32k"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 8192
TOP_P = 0.8
TOP_K = 40

SUMMARIZE_PROMPT = """Please create a summary for the TEXT TO SUMMARIZE.

Combine all of the points under the followign sections

Commonalities in DOCUMENT 1 and DOCUMENT 2
Contradictions in DOCUMENT 1 and DOCUMENT 2

Exclude any points that state that something is in one document but not the other.

TEXT TO SUMMARIZE:
{text}

OUTPUT: 
"""

COMPARE_PROMPT = """Please compare DOCUMENT 1 to DOCUMENT 2 to list commonalities and contradictions.  These documents are portions of two larger distinct documents.  
Summarize your findings in bulleted lists with your output organized in the following sections:

Commonalities in DOCUMENT 1 and DOCUMENT 2
Contradictions in DOCUMENT 1 and DOCUMENT 2

DOCUMENT 1: 
{doc1}

DOCUMENT 2: 
{doc2}

OUTPUT:
"""

def run_prompt(prompt):
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

def compare_text(doc1, doc2):
    compare_prompt = st.session_state.compare_prompt
    prompt = compare_prompt.format(doc1=doc1, doc2=doc2)
    return run_prompt(prompt)

def summarize(text):
    summarize_prompt = st.session_state.summarize_prompt
    prompt = summarize_prompt.format(text=text)
    return run_prompt(prompt)

def get_file_size(link):
    # Get file size from cloud storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.get_blob(link.replace(f"gs://{BUCKET}",""))
    return blob.size

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
    for page in reader.pages:

        # if length of chunk is greater than 30000 characters, then add it to the list
        if len(chunk) > 30000:
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


st.title('Document Compare Helper')
question=st.text_input("What sort of document are you looking for?")

def search():
    ss["search_response"] = search_sample(PROJECT, "global", DATASTORE, question)
    ss["search_df"], ss["search_meta"] = response_to_df(ss["search_response"])

if question:
    search()

def back_to_results():
    ss.doc1 = None
    ss.doc2 = None

with st.sidebar:
    st.markdown("# Settings")
    
    summarize_prompt = st.text_area("Summarize Prompt", value=SUMMARIZE_PROMPT)
    st.session_state.summarize_prompt = summarize_prompt

    compare_prompt = st.text_area("Compare Prompt", value=COMPARE_PROMPT)
    st.session_state.compare_prompt = compare_prompt

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

def show_file_selection():
    return "search_response" in ss \
        and ("doc1" not in ss or ss["doc1"] is None) \
        and ("doc2" not in ss or ss["doc2"] is  None)

if show_file_selection():
    df = ss["search_df"]
    meta = ss["search_meta"]

    with st.expander("See results", expanded=False):
        st.dataframe(df)

    rows = [row for row in df.itertuples()]
    with st.form("file_selection"):
        left, right = st.columns(2)
        with left:
            st.radio("First document to compare", 
                        map(lambda x: x.Link, rows),
                        key="doc1",
                        # captions=map(lambda x: str(get_file_size(x.Link)/(1024*1024)) + " MB", rows),
                        index=None,)
        
        with right:
            st.radio("Second document to compare", 
                        map(lambda x: x.Link, rows),
                        key="doc2",
                        # captions=map(lambda x: str(get_file_size(x.Link)/(1024*1024)) + " MB", rows),
                        index=None,)
        st.form_submit_button("Compare")

def show_analysis():
    return ("doc1" in ss and ss["doc1"] is not None) and ("doc2" in ss and ss["doc2"] is not None)

if show_analysis():
    st.button( "Back", on_click=back_to_results )

    doc1_chunks = []
    doc2_chunks = []
    
    with st.spinner("Loading documents..."):

        doc1_chunks = get_file_content(st.session_state.doc1)
        doc2_chunks = get_file_content(st.session_state.doc2)

        st.markdown(f"# {st.session_state.doc1} - {st.session_state.doc2}")
        left, right = st.columns(2)
        with left:
            st.write(f"Doc: {st.session_state.doc1}")
            st.write(f"Chunks: {len(doc1_chunks)}")
        
        with right:
            st.write(f"Doc: {st.session_state.doc2}")
            st.write(f"Chunks: {len(doc2_chunks)}")

        SLEEP_TIMEOUT = 5
    
    paragraphs = []

    breakdown, summary = st.tabs(["Page Breakdown", "Summary"])

    doc1_chunk_count = 1
    with breakdown:
        for chunk1 in doc1_chunks:
            doc2_chunk_count = 1
            text1 = chunk1[0]

            for chunk2 in doc2_chunks:

                with st.spinner(f"Comparing document 1 chunk #{doc1_chunk_count} - doc 2 chunk #{doc2_chunk_count}"):
                    text2 = chunk2[0]
                    paragraph = compare_text(text1, text2)
                    paragraphs.append(paragraph)
                    st.write( "Document 1 Pages: {first} - {last}".format(first=str(chunk1[1][0]), last=str(chunk1[1][-1])) )
                    st.write( "Document 2 Pages: {first} - {last}".format(first=str(chunk2[1][0]), last=str(chunk2[1][-1])) )
                    st.write(paragraph)
                    st.divider()
                    time.sleep(SLEEP_TIMEOUT)
            
                doc2_chunk_count += 1
            doc1_chunk_count += 1
    
    with summary:
        with st.spinner("Analyzing..."):
            response = summarize("\n\n".join(paragraphs))
            st.write(response)
