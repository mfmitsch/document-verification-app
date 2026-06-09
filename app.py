import streamlit as st

st.title("Progressive Verification Prototype")
st.write("Environment is live!")

name_input = st.sidebar.text_input("Expected Customer Name")
uploaded_file = st.file_uploader("Upload Document")

if uploaded_file:
    st.success("Document uploaded successfully!")
