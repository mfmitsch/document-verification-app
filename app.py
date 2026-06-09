import json
import uuid
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
if "custom_check_row_ids" not in st.session_state:
    st.session_state.custom_check_row_ids = []


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


class ExtractedField(BaseModel):
    field_name: str = Field(description="The name of the requested field to extract.")
    value: Optional[str] = Field(description="The value extracted from the document for this field.")


class UnderwritingExtraction(BaseModel):
    is_valid_proof_of_address: bool = Field(description="True if the document is an official utility bill, bank statement, or government mail.")
    document_type: str = Field(description="The specific type of document, e.g., 'Electric Bill', 'Bank Statement', 'Driver License', 'Invalid'.")
    extracted_name: Optional[str] = Field(description="The primary full name found on the document.")
    extracted_address: Optional[str] = Field(description="The complete mailing address found on the document.")
    additional_fields: Optional[list[ExtractedField]] = Field(
        default=None,
        description="Additional requested fields extracted from the document.",
    )
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0 based on document legibility.")
    reasoning: str = Field(description="Brief 1-sentence explanation of the document classification or validity.")


def add_custom_check_row() -> None:
    st.session_state.custom_check_row_ids.append(uuid.uuid4().hex)


def remove_custom_check_row(row_id: str) -> None:
    st.session_state.custom_check_row_ids = [
        rid for rid in st.session_state.custom_check_row_ids if rid != row_id
    ]
    st.session_state.pop(f"custom_field_{row_id}", None)
    st.session_state.pop(f"custom_expected_{row_id}", None)


def render_custom_checks_sidebar() -> list[dict[str, str]]:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="column"]
        div[data-testid="stVerticalBlock"]:has(button[kind="tertiary"]) {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.75rem;
        }
        section[data-testid="stSidebar"] button[kind="tertiary"] {
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
            padding: 0.25rem !important;
        }
        section[data-testid="stSidebar"] button[kind="tertiary"]:hover {
            background: rgba(128, 128, 128, 0.15) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("➕", help="Add verification rule", key="add_custom_check"):
        add_custom_check_row()
        st.rerun()

    for row_id in st.session_state.custom_check_row_ids:
        field_col, expected_col, delete_col = st.columns([5, 5, 1])
        with field_col:
            st.text_input(
                "Field",
                placeholder="Field name, e.g. Policy Number",
                key=f"custom_field_{row_id}",
                label_visibility="collapsed",
            )
        with expected_col:
            st.text_input(
                "Expected value",
                placeholder="Expected value",
                key=f"custom_expected_{row_id}",
                label_visibility="collapsed",
            )
        with delete_col:
            st.button(
                "🗑️",
                key=f"delete_custom_{row_id}",
                help="Remove rule",
                type="tertiary",
                on_click=remove_custom_check_row,
                args=(row_id,),
            )

    checks = []
    for row_id in st.session_state.custom_check_row_ids:
        field = st.session_state.get(f"custom_field_{row_id}", "").strip()
        expected = st.session_state.get(f"custom_expected_{row_id}", "").strip()
        if field and expected:
            checks.append({"field": field, "expected": expected})
    return checks


def build_full_prompt(base_prompt: str, custom_checks: list[dict[str, str]]) -> str:
    prompt = base_prompt.strip()
    if custom_checks:
        field_names = ", ".join(f'"{c["field"]}"' for c in custom_checks)
        prompt += (
            f"\n\nAlso extract these additional fields and return them in additional_fields "
            f"as a list of objects with field_name and value for: {field_names}."
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


def normalize_additional_fields(raw: list | dict | None) -> dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if value is not None}

    normalized: dict[str, str] = {}
    for item in raw:
        if isinstance(item, dict):
            name = item.get("field_name") or item.get("name")
            value = item.get("value")
        else:
            name = getattr(item, "field_name", None)
            value = getattr(item, "value", None)
        if name:
            normalized[str(name)] = value or ""
    return normalized


def evaluate_custom_checks(
    custom_checks: list[dict[str, str]],
    additional_fields: list | dict | None,
) -> list[dict[str, str | bool | None]]:
    extracted = normalize_additional_fields(additional_fields)
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

    failed_labels = []
    if failed_core:
        failed_labels.extend(str(f) for f in failed_core)
    if failed_custom:
        failed_labels.extend(str(f) for f in failed_custom)

    if failed_labels:
        fields = ", ".join(failed_labels)
        return f"⚠️ FLAGGED: {fields}", st.warning

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
with st.sidebar:
    custom_checks = render_custom_checks_sidebar()

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
