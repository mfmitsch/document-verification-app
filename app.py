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

DEFAULT_PROMPT = (
    "Analyze this document image for underwriting verification. "
    "Extract the primary name, mailing address, and classify what kind of document it is."
)

st.set_page_config(page_title="IDP Verification Demo", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = DEFAULT_PROMPT
if "show_prompt_editor" not in st.session_state:
    st.session_state.show_prompt_editor = False
if "custom_checks_df" not in st.session_state:
    st.session_state.custom_checks_df = pd.DataFrame(
        {"Field": pd.Series(dtype="str"), "Expected Value": pd.Series(dtype="str")}
    )


def require_auth() -> None:
    if st.session_state.authenticated:
        return

    st.title("🚗 Document Verification Portal")
    st.caption("Intelligent Document Processing (IDP) Pipeline")

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
    additional_fields: Optional[dict[str, str]] = Field(
        default=None,
        description="Map of additionally requested field names to values extracted from the document.",
    )
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0 based on document legibility.")
    reasoning: str = Field(description="Brief 1-sentence explanation of the document classification or validity.")


def parse_custom_checks(df: pd.DataFrame) -> list[dict[str, str]]:
    checks = []
    for _, row in df.iterrows():
        field = str(row.get("Field", "")).strip()
        expected = str(row.get("Expected Value", "")).strip()
        if field and expected:
            checks.append({"field": field, "expected": expected})
    return checks


def build_full_prompt(base_prompt: str, custom_checks: list[dict[str, str]]) -> str:
    prompt = base_prompt.strip()
    if custom_checks:
        field_names = ", ".join(f'"{c["field"]}"' for c in custom_checks)
        prompt += (
            f"\n\nAlso extract these additional fields and return them in additional_fields "
            f"using these exact keys: {field_names}."
        )
    return prompt


def evaluate_name_match(expected_name: str, extracted_name: str | None) -> bool:
    last_name = expected_name.split()[-1].lower()
    return last_name in (extracted_name or "").lower()


def evaluate_address_match(expected_address: str, extracted_address: str | None) -> bool:
    extracted = (extracted_address or "").lower()
    expected = expected_address.lower()
    if not extracted or not expected:
        return False
    if expected in extracted:
        return True
    tokens = [
        token.strip(".,#")
        for token in expected.replace(",", " ").split()
        if len(token.strip(".,#")) >= 3
    ]
    if not tokens:
        return False
    matched = sum(1 for token in tokens if token in extracted)
    return matched >= min(2, len(tokens))


def evaluate_core_checks(
    expected_name: str,
    expected_address: str,
    extracted_name: str | None,
    extracted_address: str | None,
) -> list[dict[str, str | bool | None]]:
    name_match = evaluate_name_match(expected_name, extracted_name)
    address_match = evaluate_address_match(expected_address, extracted_address)
    return [
        {
            "field": "Name",
            "expected": expected_name,
            "extracted": extracted_name,
            "match": name_match,
        },
        {
            "field": "Address",
            "expected": expected_address,
            "extracted": extracted_address,
            "match": address_match,
        },
    ]


def evaluate_custom_checks(
    custom_checks: list[dict[str, str]],
    additional_fields: dict[str, str] | None,
) -> list[dict[str, str | bool | None]]:
    extracted = additional_fields or {}
    normalized = {key.lower(): value for key, value in extracted.items()}
    results = []

    for check in custom_checks:
        field = check["field"]
        expected = check["expected"]
        extracted_value = extracted.get(field) or normalized.get(field.lower())
        match = (
            expected.lower() in extracted_value.lower()
            if extracted_value
            else False
        )
        results.append({
            "field": field,
            "expected": expected,
            "extracted": extracted_value,
            "match": match,
        })

    return results


def determine_status(
    doc_type_valid: bool,
    core_results: list[dict[str, str | bool | None]],
    custom_results: list[dict[str, str | bool | None]],
) -> tuple[str, callable]:
    failed_core = [r["field"] for r in core_results if not r["match"]]
    failed_custom = [r["field"] for r in custom_results if not r["match"]]

    if not doc_type_valid:
        return "❌ REJECTED: INVALID DOC", st.error
    if failed_core:
        fields = " & ".join(str(f).upper() for f in failed_core)
        return f"⚠️ FLAGGED: {fields} MISMATCH", st.warning
    if failed_custom:
        fields = ", ".join(str(f) for f in failed_custom)
        return f"⚠️ FLAGGED: CHECK FAILED ({fields})", st.warning
    return "✅ APPROVED", st.success


require_auth()

st.title("🚗 Document Verification Portal")
st.caption("Intelligent Document Processing (IDP) Pipeline")

st.sidebar.markdown("### 🗄️ Required Verification Rules")
st.sidebar.caption("Name and address are always checked against the document.")
expected_name = st.sidebar.text_input("Expected Account Name", "Jane M. Doe")
expected_address = st.sidebar.text_input("Expected Risk Address", "123 Main St, Indianapolis, IN")

st.sidebar.markdown("### ➕ Additional Verification Rules")
st.sidebar.caption("Optional checks beyond name and address.")
custom_checks_df = st.sidebar.data_editor(
    st.session_state.custom_checks_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Field": st.column_config.TextColumn("Field", help="Label to extract from the document, e.g. Policy Number"),
        "Expected Value": st.column_config.TextColumn("Expected Value", help="Value to match against the extraction"),
    },
    hide_index=True,
)
st.session_state.custom_checks_df = custom_checks_df
custom_checks = parse_custom_checks(custom_checks_df)

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

    prompt_col, prompt_btn_col = st.columns([3, 1])
    with prompt_btn_col:
        if st.button("View / Edit Prompt", use_container_width=True):
            st.session_state.show_prompt_editor = not st.session_state.show_prompt_editor

    if st.session_state.show_prompt_editor:
        prompt_col1, prompt_col2 = st.columns([4, 1])
        with prompt_col1:
            st.session_state.user_prompt = st.text_area(
                "Model instructions",
                value=st.session_state.user_prompt,
                height=120,
                help="Edit the base instructions sent to the model. Additional field extraction is appended automatically.",
            )
        with prompt_col2:
            if st.button("Reset prompt", use_container_width=True):
                st.session_state.user_prompt = DEFAULT_PROMPT
                st.rerun()

        full_prompt_preview = build_full_prompt(st.session_state.user_prompt, custom_checks)
        st.caption("Full prompt sent to the model:")
        st.code(full_prompt_preview)

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
                    full_prompt = build_full_prompt(st.session_state.user_prompt, custom_checks)

                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[image, full_prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=UnderwritingExtraction,
                            temperature=0.1,
                        ),
                    )

                    result_data = json.loads(response.text)
                    st.session_state.scan_count += 1

                    extracted_name_raw = result_data.get("extracted_name") or ""
                    extracted_address_raw = result_data.get("extracted_address") or ""
                    doc_type_valid = result_data.get("is_valid_proof_of_address", False)

                    core_results = evaluate_core_checks(
                        expected_name,
                        expected_address,
                        extracted_name_raw,
                        extracted_address_raw,
                    )
                    custom_results = evaluate_custom_checks(
                        custom_checks,
                        result_data.get("additional_fields"),
                    )

                    status, color_box = determine_status(doc_type_valid, core_results, custom_results)

                    color_box(f"**Verification Status: {status}**")

                    st.markdown("#### Extracted Entity Insights")
                    st.json(result_data)

                    st.markdown("#### Verification Rule Results")
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Field": r["field"],
                                "Expected": r["expected"],
                                "Extracted": r["extracted"] or "—",
                                "Match": "✅" if r["match"] else "❌",
                            }
                            for r in core_results + custom_results
                        ]),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.session_state.scan_history.append({
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Doc Type": result_data.get("document_type"),
                        "Extracted Name": extracted_name_raw,
                        "Status": status,
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
