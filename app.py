from smart_compliance.kyc.core import KycPhoto, Kyc
from smart_compliance.kyc.detector import verify_face
import streamlit as st
import numpy as np
import pandas as pd
import cv2

kyc = Kyc()

st.session_state = {
    "models": [],
    "similarity_metrics": [],
}

models = [
    "VGG-Face",
    "Facenet",
    "OpenFace",
    "ArcFace",
]

similarity_metrics = [
    "cosine",
    "euclidean",
    "euclidean_l2"
]


def models_form():
    with st.sidebar.expander("Step 1: Choose your models"):
        st.session_state["models"] = st.multiselect(
            "Trained models to detect and predict", options=models, default=models, key="models")


def similarity_metrics_form():
    with st.sidebar.expander("Step 2: Choose your similarity metrics"):
        st.session_state["similarity_metrics"] = st.multiselect(
            "Metrics used to determine similarities", options=similarity_metrics, default=similarity_metrics, key="similarity_metrics")


def selfie_form():
    with st.sidebar.expander("Step 3: Upload your selfie"):
        st.write("Your face should be clearly visible. Remove masks, glasses and hats. Look straight at the camera. Have a neutral face matching your face on your legal document. Make sure the lighting is good. Avoid shadows and busy backgrounds.")

        media_type = st.selectbox(
            "How will you upload your selfie?", ("", "Camera/Webcam", "Upload"), key="selfie_media_type")

        if (media_type == ""):
            return

        selfie = None

        if media_type == "Camera/Webcam":
            selfie = st.camera_input(
                "Upload your photo", key="selfie_photo")

        if media_type == "Upload":
            selfie = st.file_uploader(
                "Upload your photo", type=['jpg', 'png', 'jpeg'], accept_multiple_files=False, key="selfie_upload")

        if selfie != None:
            image = cv2.imdecode(np.fromstring(
                selfie.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            kyc.register_selfie(
                KycPhoto(image))

    with st.expander("Selfie"):
        col1, col2 = st.columns(2)
        if kyc.selfie == None:
            st.error("No selfie uploaded yet.")
            return

        with col1:
            st.image(
                kyc.selfie.original, caption=f"Original image {kyc.selfie.original.shape}", use_column_width=True)

        with col2:
            st.image(
                kyc.selfie.detected_faces, caption=f"Processed image {kyc.selfie.detected_faces.shape}", use_column_width=True)

        st.divider()

        if kyc.selfie != None:
            models = st.session_state["models"]
            similarity_metrics = st.session_state["similarity_metrics"]
            results = []
            for i, face in enumerate(kyc.selfie.faces):
                results = pd.DataFrame(verify_face(
                    kyc.selfie.base_image, face, models, similarity_metrics))

                st.data_editor(
                    results,
                    key=f"selfie_results_{i}",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "base": st.column_config.ImageColumn("Base", help="Base image"),
                        "face": st.column_config.ImageColumn("Face", help="Detected face"),
                        "verified": st.column_config.CheckboxColumn("Verified", help="Indicates if the similarity is high enough to be a match"),
                        "model": st.column_config.TextColumn("Model", help="Model used to detect and predict"),
                        "similarity_metric": st.column_config.TextColumn("Similarity metric", help="Model used to detect and predict"),
                        "distance": st.column_config.NumberColumn("Distance", help="Distance between the two faces"),
                        "threshold": st.column_config.NumberColumn("Threshold", help="Threshold used to determine if the two faces are the same"),
                    }
                )


def document_form():
    with st.sidebar.expander("Step 4: Upload your legal document"):

        if kyc.selfie == None:
            st.write("You need to upload the selfie first.")
            return

        st.write("Text should be clearly visible.")

        media_type = st.selectbox(
            "How will you upload your document?", ("", "Camera/Webcam", "Upload"), key="doc_media_type")

        if (media_type == ""):
            return

        document = None

        if media_type == "Camera/Webcam":
            document = st.camera_input(
                "Upload your document", key="doc_photo")

        if media_type == "Upload":
            document = st.file_uploader(
                "Upload your document", type=['jpg', 'png', 'jpeg'], accept_multiple_files=False, key="doc_upload")

        if document != None:
            image = cv2.imdecode(np.fromstring(
                document.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            kyc.register_document(
                KycPhoto(image))

    with st.expander("Legal document"):
        col1, col2 = st.columns(2)
        if kyc.document == None:
            st.error("No legal document uploaded yet.")
            return

        with col1:
            st.image(
                kyc.document.original, caption=f"Original image {kyc.document.original.shape}", use_column_width=True)

        with col2:
            st.image(
                kyc.document.detected_faces, caption=f"Processed image {kyc.document.detected_faces.shape}", use_column_width=True)

        st.divider()

        if kyc.document != None:
            models = st.session_state["models"]
            similarity_metrics = st.session_state["similarity_metrics"]
            results = []
            for i, face in enumerate(kyc.document.faces):
                results = pd.DataFrame(verify_face(
                    kyc.document.base_image, face, models, similarity_metrics)).sort_values(by=['verified'], ascending=False)
                st.data_editor(
                    results,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "base": st.column_config.ImageColumn("Base", help="Base image"),
                        "face": st.column_config.ImageColumn("Face", help="Detected face"),
                        "verified": st.column_config.CheckboxColumn("Verified", help="Indicates if the similarity is high enough to be a match"),
                        "model": st.column_config.TextColumn("Model", help="Model used to detect and predict"),
                        "similarity_metric": st.column_config.TextColumn("Similarity metric", help="Model used to detect and predict"),
                        "distance": st.column_config.NumberColumn("Distance", help="Distance between the two faces"),
                        "threshold": st.column_config.NumberColumn("Threshold", help="Threshold used to determine if the two faces are the same"),
                    }
                )


def verification_form():
    if (kyc.selfie == None or kyc.document == None):
        return

    with st.expander("Verification"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                kyc.selfie.base_image, caption=f"Selfie", use_column_width=True)

        with col2:
            st.image(
                kyc.document.base_image, caption=f"Document", use_column_width=True)

        results = pd.DataFrame(verify_face(
            kyc.selfie.base_image, kyc.document.base_image, models, similarity_metrics)).sort_values(by=['verified'], ascending=False)
        st.data_editor(
            results,
            hide_index=True,
            use_container_width=True,
            column_config={
                "base": st.column_config.ImageColumn("Base", help="Base image"),
                "face": st.column_config.ImageColumn("Face", help="Detected face"),
                "verified": st.column_config.CheckboxColumn("Verified", help="Indicates if the similarity is high enough to be a match", disabled=True),
                "model": st.column_config.TextColumn("Model", help="Model used to detect and predict"),
                "similarity_metric": st.column_config.TextColumn("Similarity metric", help="Model used to detect and predict"),
                "distance": st.column_config.NumberColumn("Distance", help="Distance between the two faces"),
                "threshold": st.column_config.NumberColumn("Threshold", help="Threshold used to determine if the two faces are the same"),
            }
        )


def details_form():
    with st.sidebar.expander("Step 5: Enter your details"):
        with st.form("compliance_form"):
            first_name = st.text_input('First name')
            last_name = st.text_input('Last name')
            id_number = st.text_input('ID number / passport / NIC')
            dob = st.date_input('Date of birth')
            st.form_submit_button(
                "Submit", use_container_width=True, type="primary")


def main():
    st.set_page_config(
        page_title="Smart Compliance", page_icon="✨")

    st.title("Smart Compliance ✨")
    st.write(
        "A fraud reduction strategy for financial institutions done in a few easy steps.")

    st.sidebar.title("Compliance documents")

    models_form()
    similarity_metrics_form()
    selfie_form()
    document_form()
    verification_form()
    details_form()

    st.sidebar.divider()
    st.sidebar.image("./hero.png", use_column_width=True,
                     caption="Smart Compliance - Le Wagon Project for Batch #1287 Mauritius with Clarice Bouwer, Amit Malik and Vighnesh Gaya.")


if __name__ == "__main__":
    main()

# https://image-coordinates.streamlit.app/
