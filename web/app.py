import streamlit as st

with st.form("compliance_form"):
    first_name = st.text_input('First name')
    last_name = st.text_input('Last name')
    id_number = st.text_input('ID number / passport / NIC')
    dob = st.date_input('Date of birth')
    picture = st.camera_input("Take a picture")

    if picture:
        st.image(picture)

    submitted = st.form_submit_button("Submit")
