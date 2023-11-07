import streamlit as st

with st.form("compliance_form"):
    first_name = st.text_input('First name')
    last_name = st.text_input('Last name')
    id_number = st.text_input('ID number / passport / NIC')
    dob = st.date_input('Date of birth')
    photo = st.camera_input("Take a picture")
    fallback_photo = st.file_uploader("Upload a photo if you can't take one", type=["png", "jpg", "jpeg"])

    if photo:
        st.image(photo)

    submitted = st.form_submit_button("Submit")
