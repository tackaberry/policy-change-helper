# import libraries
import streamlit as st
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


parameters = {
    "candidate_count": 1,
    "temperature": 0.2,
    "max_output_tokens": 256,
    "top_p": 0.8,
    "top_k": 40
}

NOT_FOUND = "1234"
def get_paragraphs(policy, change):
    prompt = f"""
    You are an intelligent policy analyst focused on determining if a PROPOSED CHANGE has any impact on a provided POLICY.  
    Please respond with RELEVANT TEXT extracted from the POLICY verbatim.
    Only respond with text that has a demonstrated link to the PROPOSED CHANGE.
    Respond with {NOT_FOUND} if no RELEVANT TEXT can be found.

    PROPOSED CHANGE: {change}

    POLICY: {policy}
    RELEVANT TEXT: 
    """

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        prompt,
        **parameters
    )
    return response.text

def get_text(policy, proposal):
    prompt = f"""
    You are an intelligent policy analyst helping on determine the IMPACT of a PROPOSED CHANGE on an an EXISTING POLICY.
    Please quote the relevant portion of the EXISTING POLICY when explaining the IMPACT.
    Strictly Use ONLY the following pieces of context to determine the IMPACT of the PROPOSED CHANGE.

    EXISTING POLICY: 
    {policy}

    PROPOSED CHANGE:
    {proposal}

    IMPACT:

    """

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        prompt,
        **parameters
    )
    return response.text

def get_file_content(link):
    # download file from google cloud storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET)
    blob = bucket.blob(link.replace(f"gs://{BUCKET}",""))
    blob.download_to_filename("temp.pdf")
    reader = PdfReader('temp.pdf') 

    chunks = []
    chunk = ""
    for page in reader.pages:

        # if length of chunk is greater than 1000 characters, then add it to the list
        if len(chunk) > 20000:
            chunks.append(chunk)
            chunk = ""

        chunk += page.extract_text()

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
        # For information about search summaries, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/get-search-summaries
        summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
            summary_result_count=5,
            include_citations=True,
            ignore_adversarial_query=True,
            ignore_non_summary_seeking_query=True,
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
        "summary": firstPage["summary"]["summaryText"],
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

if not question:
    st.warning("Please enter a question.")

if question:
    if st.session_state.showTwo:
        st.session_state.showOne = False
    else:
        st.session_state.showOne = True


def analyze_this(link, question):
    st.session_state.link = link
    st.session_state.showOne = False
    st.session_state.showTwo = True


if st.session_state.showOne:

    response2 = search_sample(PROJECT, "global", DATASTORE, question)
    [df, meta] = response_to_df(response2)

    with st.expander("See results", expanded=False):
        st.dataframe(df)

    st.write(meta["summary"])

    for row in df.itertuples():
        st.markdown(f"# {row.Title}")
        st.markdown(row.Snippet, unsafe_allow_html=True)
        st.button("Analyze this", key=row.Id, on_click=analyze_this, args=(row.Link, question))
        st.divider()
        
if st.session_state.showTwo:

    st.markdown(f"# {st.session_state.link}")

    st.write(st.session_state.link)

    params = st.experimental_get_query_params()

    pages = get_file_content(st.session_state.link)

    st.write(f"Doc: {st.session_state.link}")
    st.write(f"Pages: {len(pages)}")

    SLEEP_TIMEOUT = 5
    paragraphs = []
    for page in pages:
        paragraph = get_paragraphs(page, question)
        if (not paragraph.startswith(NOT_FOUND)):
            paragraphs.append(paragraph)
        st.write(paragraph)
        time.sleep(SLEEP_TIMEOUT)

    response = get_text("\n\n".join(paragraphs), question)
    st.write(response)


    # how will this policy change if we remove horns from boats