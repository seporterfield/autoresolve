import streamlit as st
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()


def summarize(input_string: str) -> str:
    llm = OpenAI(temperature=0)  # type: ignore[call-arg]
    text_splitter = CharacterTextSplitter()
    contract = input_string
    texts = text_splitter.split_text(contract)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain.run(docs)


input_string = st.text_area("Please enter your text here:", "")
if run_button := st.button("Run"):
    output_string = summarize(input_string)
    st.text("The summarized output is:")
    st.write(output_string)
