import json
import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

MAX_SCANS_PER_SESSION = 3

st.set_page_config(page_title="IDP Verification Demo", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


def require_auth() -> None:
    if st.session_state.authenticated:
        return

    st.title("🚗 Auto Underwriting: Document Verification Portal")
    st.caption("Lead Data Science Portfolio Piece - Intelligent Document Processing (IDP) Pipeline")

    if "APP_PASSWORD" not in st.secrets:
        st.error("App password is not configured. Set APP_PASSWORD in Streamlit secrets.")
        st.stop()

    st.text_input(
        "Password",
        type="password",
        key="login_password",
        help="Reach out to me for the password.",
    )
    if st.button("Unlock demo"):
        if st.session_state.login_password == st.secrets["APP_PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


class UnderwritingExtraction(BaseModel):
    is_valid_proof_of_address: bool = Field(description="True if the document is an official utility bill, bank statement, or government mail.")
    document_type: str = Field(description="The specific type of document, e.g., 'Electric Bill', 'Bank Statement', 'Driver License', 'Invalid'.")
    extracted_name: Optional[str] = Field(description="The primary full name found on the document.")
    extracted_address: Optional[str] = Field(description="The complete mailing address found on the document.")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0 based on document legibility.")
    reasoning: str = Field(description="Brief 1-sentence explanation of the document classification or validity.")


require_auth()

st.title("🚗 Auto Underwriting: Document Verification Portal")
st.caption("Lead Data Science Portfolio Piece - Intelligent Document Processing (IDP) Pipeline")

st.sidebar.markdown("### 🗄️ Mock Customer Database Record")
expected_name = st.sidebar.text_input("Expected Account Name", "Jane M. Doe")
expected_address = st.sidebar.text_input("Expected Risk Address", "123 Main St, Indianapolis, IN")

remaining_scans = MAX_SCANS_PER_SESSION - st.session_state.scan_count
st.sidebar.caption(f"Scans remaining this session: {max(remaining_scans, 0)} / {MAX_SCANS_PER_SESSION}")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 Upload Proof of Address")
    uploaded_file = st.file_uploader("Upload an image of a utility bill, statement, or ID", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document Preview", use_container_width=True)

with col2:
    st.markdown("### 🤖 Automated AI Determination")

    if uploaded_file and st.button("⚡ Run Underwriting Rules Engine"):
        if st.session_state.scan_count >= MAX_SCANS_PER_SESSION:
            st.error(
                f"Session limit reached ({MAX_SCANS_PER_SESSION} scans). "
                "Refresh the page to start a new session."
            )
        elif "GOOGLE_API_KEY" not in st.secrets:
            st.error("API key is not configured. Set GOOGLE_API_KEY in Streamlit secrets.")
        else:
            with st.spinner("Processing image through multimodal layout model..."):
                try:
                    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            image,
                            "Analyze this document image for underwriting verification. Extract the primary name, mailing address, and classify what kind of document it is."
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=UnderwritingExtraction,
                            temperature=0.1,
                        ),
                    )

                    result_data = json.loads(response.text)
                    st.session_state.scan_count += 1

                    last_name_expected = expected_name.split()[-1].lower()
                    extracted_name_raw = result_data.get("extracted_name") or ""

                    name_match = last_name_expected in extracted_name_raw.lower()
                    doc_type_valid = result_data.get("is_valid_proof_of_address", False)

                    if doc_type_valid and name_match:
                        status = "✅ APPROVED"
                        color_box = st.success
                    elif doc_type_valid and not name_match:
                        status = "⚠️ FLAGGED: NAME MISMATCH"
                        color_box = st.warning
                    else:
                        status = "❌ REJECTED: INVALID DOC"
                        color_box = st.error

                    color_box(f"**Verification Status: {status}**")

                    st.markdown("#### Extracted Entity Insights")
                    st.json(result_data)

                    st.session_state.scan_history.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Doc Type": result_data.get("document_type"),
                        "Extracted Name": extracted_name_raw,
                        "Status": status
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("### 📜 Real-Time Audit Log (Current Underwriting Session)")
if st.session_state.scan_history:
    st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True)
    if st.button("Clear Audit Trail"):
        st.session_state.scan_history = []
        st.rerun()
else:
    st.info("Awaiting document upload and rules execution.")
