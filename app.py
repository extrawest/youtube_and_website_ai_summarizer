import re

import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi


def get_youtube_id(url):
    """Extract video ID from YouTube URL"""
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None


def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([entry['text'] for entry in transcript_list])
        return transcript_text
    except Exception as e:
        st.error(f"Error getting YouTube transcript: {str(e)}")
        st.error("Make sure the video has closed captions available.")
        return None


st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

groq_api_key = st.text_input("Groq API Key", value="", type="password")

if groq_api_key:
    generic_url = st.text_input("URL", label_visibility="collapsed")

    llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

    prompt_template = """
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    if st.button("Summarize the Content from YT or Website"):
        if not groq_api_key.strip() or not generic_url.strip():
            st.error("Please provide the information to get started")
        elif not validators.url(generic_url):
            st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
        else:
            try:
                with st.spinner("Processing content..."):
                    if any(url in generic_url for url in ["youtube.com", "youtu.be"]):
                        video_id = get_youtube_id(generic_url)
                        if not video_id:
                            st.error("Could not extract YouTube video ID from URL")
                            st.stop()

                        transcript = get_youtube_transcript(video_id)
                        if not transcript:
                            st.stop()

                        # Create a Document object from the transcript
                        docs = [Document(page_content=transcript)]
                    else:
                        loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                       headers={
                                                           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                                                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                                         "Chrome/116.0.0.0 Safari/537.36"})
                        docs = loader.load()

                    with st.spinner("Generating summary..."):
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.invoke(docs)["output_text"]
                        st.success(output_summary)

            except Exception as e:
                st.exception(f"An error occurred: {str(e)}")
