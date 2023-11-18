from smart_compliance.kyc.core import KycPhoto, Kyc
from smart_compliance.kyc.detector import verify_face
import streamlit as st
import numpy as np
import cv2

kyc = Kyc()

st.session_state = {
    "models": [],
    "similarity_metrics": [],
}


def models_form():
    with st.sidebar.expander("Step 1: Choose your models"):
        st.session_state["models"] = st.multiselect("Choose your models", [
            "Facenet",
            "OpenFace",
            "ArcFace",
        ], key="models")


def similarity_metrics_form():
    with st.sidebar.expander("Step 2: Choose your similarity metrics"):
        st.session_state["similarity_metrics"] = st.multiselect(
            "Choose your distance metrics", [
                "cosine",
                "euclidean",
                "euclidean_l2"
            ], key="similarity_metrics")


def selfie_form():
    with st.sidebar.expander("Step 3: Upload your selfie"):
        st.write("Your face should be clearly visible. Remove masks, glasses and hats. Look straight at the camera. Have a neutral face matching your face on your legal document. Make sure the lighting is good. Avoid shadows and busy backgrounds.")

        media_type = st.selectbox(
            "How will you upload your selfie?", ("", "Camera/Webcam", "Upload"), key="selfie_media_type")

        if (media_type == ""):
            return

        scale_factor = st.slider(
            "Scale the image by a factor of", min_value=1.0, max_value=3.0, value=1.1, step=0.1, key="selfie_scale_factor")

        padding = st.slider(
            "Pad image by pixels", min_value=0, max_value=200, value=50, step=10, key="selfie_padding_pixels")

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
                KycPhoto(image, scale_factor=scale_factor, padding=padding))

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

            cols_faces = st.columns(len(kyc.selfie.faces) + 1)
            cols_results = st.columns(len(kyc.selfie.faces) + 1)
            cols_faces[0].image(
                kyc.base_image, caption=f"Base image {kyc.base_image.shape}", use_column_width=True)
            cols_results[0].write(verify_face(
                kyc.base_image, kyc.base_image, models, similarity_metrics))
            for i, face in enumerate(kyc.selfie.faces):
                cols_faces[i+1].image(face,
                                      caption=f"Face {i+1} {face.shape}", use_column_width=True)
                cols_results[i+1].write(verify_face(kyc.base_image,
                                        face, models, similarity_metrics))


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

        scale_factor = st.slider(
            "Scale the image by a factor of", min_value=1.0, max_value=3.0, value=1.1, step=0.1, key="doc_scale_factor")

        padding = st.slider(
            "Pad image by pixels", min_value=0, max_value=200, value=50, step=10, key="doc_padding_pixels")

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
                KycPhoto(image, scale_factor=scale_factor, padding=padding))

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

            cols_faces = st.columns(len(kyc.document.faces) + 1)
            cols_results = st.columns(len(kyc.document.faces) + 1)
            cols_faces[0].image(
                kyc.base_image, caption=f"Base image {kyc.base_image.shape}", use_column_width=True)
            cols_results[0].write(verify_face(
                kyc.base_image, kyc.base_image, models, similarity_metrics))
            for i, face in enumerate(kyc.document.faces):
                cols_faces[i+1].image(face,
                                      caption=f"Face {i+1} {face.shape}", use_column_width=True)
                cols_results[i+1].write(verify_face(kyc.base_image,
                                        face, models, similarity_metrics))


def details_form():
    with st.sidebar.expander("Step 5: Enter your details"):
        with st.form("compliance_form"):
            first_name = st.text_input('First name')
            last_name = st.text_input('Last name')
            id_number = st.text_input('ID number / passport / NIC')
            dob = st.date_input('Date of birth')
            st.form_submit_button("Submit", use_container_width=True, type="primary")


def main():
    st.set_page_config(
        page_title="Smart Compliance", page_icon="✨")

    st.title("Smart Compliance ✨")
    st.write(
        "A fraud reduction strategy for financial institutions done in three easy steps.")

    st.sidebar.title("Compliance documents")

    models_form()
    similarity_metrics_form()
    selfie_form()
    document_form()
    details_form()


if __name__ == "__main__":
    main()

# https://image-coordinates.streamlit.app/