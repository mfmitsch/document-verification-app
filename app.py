import streamlit as st
import pandas as pd
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

# 1. Page Configuration & Title
st.set_page_config(page_title="Progressive IDP Prototype", layout="wide")
st.title("🚗 Progressive Underwriting: Auto-Verification Portal")
st.caption("Lead Data Science Portfolio Piece - Intelligent Document Processing (IDP) Pipeline")

# 2. Securely get the API Key from the user interface for local testing
# (When deploying to Streamlit Cloud, you can hide this in Advanced Secrets!)
api_key = st.sidebar.text_input("Enter Google AI Studio API Key", type="password")

# 3. Define the Structured Output Schema using Pydantic
class UnderwritingExtraction(BaseModel):
    is_valid_proof_of_address: bool = Field(description="True if the document is an official utility bill, bank statement, or government mail.")
    document_type: str = Field(description="The specific type of document, e.g., 'Electric Bill', 'Bank Statement', 'Driver License', 'Invalid'.")
    extracted_name: Optional[str] = Field(description="The primary full name found on the document.")
    extracted_address: Optional[str] = Field(description="The complete mailing address found on the document.")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0 based on document legibility.")
    reasoning: str = Field(description="Brief 1-sentence explanation of the document classification or validity.")

# 4. Initialize Session State for the History Log
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

# 5. UI Layout - Sidebar for Expected Database Values
st.sidebar.markdown("### 🗄️ Mock Customer Database Record")
expected_name = st.sidebar.text_input("Expected Account Name", "Jane M. Doe")
expected_address = st.sidebar.text_input("Expected Risk Address", "123 Main St, Indianapolis, IN")

# 6. Main Dashboard Layout
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
        if not api_key:
            st.error("Please enter your AI Studio API Key in the sidebar to run the live model.")
        else:
            with st.spinner("Processing image through multimodal layout model..."):
                try:
                    # Initialize the official Google GenAI client
                    client = genai.Client(api_key=api_key)
                    
                    # Direct the multimodal model to extract the data using our strict Pydantic structure
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            image,
                            "Analyze this document image for underwriting verification. Extract the primary name, mailing address, and classify what kind of document it is."
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=UnderwritingExtraction,
                            temperature=0.1 # Low temperature for highly deterministic extraction
                        ),
                    )
                    
                    # Parse the structured response back into Python
                    # The SDK parses json directly into our Pydantic object if using parsed responses,
                    # or we can read the raw string text content:
                    import json
                    result_data = json.loads(response.text)
                    
                    # --- BUSINESS RULES ENGINE ---
                    # Implement simple fuzzy logic: Check if the expected last name is in the extracted name
                    last_name_expected = expected_name.split()[-1].lower()
                    extracted_name_raw = result_data.get("extracted_name") or ""
                    
                    name_match = last_name_expected in extracted_name_raw.lower()
                    doc_type_valid = result_data.get("is_valid_proof_of_address", False)
                    
                    # Determine final rule status
                    if doc_type_valid and name_match:
                        status = "✅ APPROVED"
                        color_box = st.success
                    elif doc_type_valid and not name_match:
                        status = "⚠️ FLAGGED: NAME MISMATCH"
                        color_box = st.warning
                    else:
                        status = "❌ REJECTED: INVALID DOC"
                        color_box = st.error
                    
                    # Display Results Screen
                    color_box(f"**Verification Status: {status}**")
                    
                    st.markdown("#### Extracted Entity Insights")
                    st.json(result_data)
                    
                    # Log to history
                    st.session_state.scan_history.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Doc Type": result_data.get("document_type"),
                        "Extracted Name": extracted_name_raw,
                        "Status": status
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# 7. Permanent Audit Trail History Grid at the bottom
st.markdown("---")
st.markdown("### 📜 Real-Time Audit Log (Current Underwriting Session)")
if st.session_state.scan_history:
    st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True)
    if st.button("Clear Audit Trail"):
        st.session_state.scan_history = []
        st.rerun()
else:
    st.info("Awaiting document upload and rules execution.")
