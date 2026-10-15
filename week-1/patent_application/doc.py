"""
Patent Classification App
Powered by Amazon Bedrock (Claude Haiku 4.5)

Upload a PDF patent application and receive an IPC classification
with reasoning from an LLM.
"""

import json
import boto3
import pypdf
import streamlit as st

from config import SYSTEM_PROMPT, BEDROCK_MODEL_ID, BEDROCK_REGION


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Patent Classifier",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Patent Application Classifier")
st.markdown(
    "Upload a patent application PDF and click **Classify** to receive an "
    "[IPC](https://www.wipo.int/classifications/ipc/en/) classification "
    "with reasoning."
)

st.divider()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    reader = pypdf.PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def classify_patent(patent_text: str) -> str:
    """
    Send the patent text to Bedrock and return the model's classification.
    Uses the instance role for authentication — no explicit credentials needed.
    """
    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": patent_text,
                }
            ],
        }
    )

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    return response_body["content"][0]["text"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Patent Application (PDF)",
    type=["pdf"],
    help="Select a PDF containing the patent application you want to classify.",
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    if st.button("🔍 Classify", type="primary", use_container_width=True):
        with st.spinner("Extracting text from PDF…"):
            try:
                patent_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read the PDF: {e}")
                st.stop()

        if not patent_text.strip():
            st.warning(
                "No readable text was found in the PDF. "
                "Make sure the document is not a scanned image-only PDF."
            )
            st.stop()

        with st.spinner("Classifying patent… this may take a moment for large documents."):
            try:
                result = classify_patent(patent_text)
            except Exception as e:
                st.error(f"Bedrock request failed: {e}")
                st.stop()

        st.divider()
        st.subheader("Classification Result")
        st.markdown(result)

else:
    st.info("👆 Please upload a PDF to get started.")
