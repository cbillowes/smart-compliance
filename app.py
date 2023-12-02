from smart_compliance.kyc.core import KycSelfie, KycDocument, Kyc
from smart_compliance.kyc.detector import verify_face, get_prediction
import imageio.v3 as iio
import streamlit as st
import numpy as np
import pandas as pd

kyc = Kyc()

st.session_state = {
    "models": [],
    "similarity_metrics": [],
}

models = [
    "VGG-Face",
    "Facenet",
    "ArcFace",
]

model_weights = [
    {"model": "VGG-Face", "weight": 10},
    {"model": "Facenet", "weight": 40},
    {"model": "ArcFace", "weight": 50},
]

similarity_metrics = [
    "euclidean_l2"
]

st.session_state["models"] = models
st.session_state["similarity_metrics"] = similarity_metrics


def info_form():
    with st.sidebar.expander("📝 Information"):
        st.write("This is a proof of concept for a fraud reduction strategy for financial institutions. It is not intended to be used in production yet.")
        st.write("The purpose of this project is to demonstrate how one can use facial recognition to verify the identity of a customer.")
        st.write("This project uses the following technologies:")
        st.write("- Python")
        st.write("- Streamlit")
        st.write("- OpenCV")
        st.write("- DeepFace")
        st.write("- Pytesseract")


def selfie_form():
    with st.expander("Upload your selfie"):
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
            image = iio.imread(selfie)
            image_array = np.array(image)
            kyc.register_selfie(KycSelfie(image_array))


        try:
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
                if kyc.selfie.error != None:
                    st.error(kyc.selfie.error)
                    return

                models = st.session_state["models"]
                similarity_metrics = st.session_state["similarity_metrics"]
                results = pd.DataFrame(verify_face(
                    kyc.selfie.base_image, kyc.selfie.face, models, similarity_metrics))

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
        except Exception as e:
            print("Could not show selfie images: " + str(e))


def document_form():
    try:
        with st.expander("Upload your legal document"):

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
                image = iio.imread(document)
                image_array = np.array(image)
                kyc.register_document(KycDocument(image_array))

                if kyc.document.error != None:
                    st.error(kyc.document.error)
                    return

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
    except Exception as e:
        print("Could not show document images: " + str(e))


def verification_form():
    if (kyc.selfie == None or kyc.document == None):
        return

    try:
        print("Verifying faces...")
        results = verify_face(
            kyc.selfie.base_image, kyc.document.base_image, models, similarity_metrics)

        predict = get_prediction(results, model_weights)
        if predict == "verified":
            st.write(f"## ✅ Verified")
        elif predict == "not_verified":
            st.write(f"## ❌ Not verified")

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                kyc.selfie.base_image, caption=f"Selfie", use_column_width=True)

        with col2:
            st.image(
                kyc.document.base_image, caption=f"Document", use_column_width=True)

        results = pd.DataFrame(results).sort_values(
            by=['verified'], ascending=False)
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

        print("Extracting text...")
        extracted_text = kyc.document.extract_text({
            "first_name": st.session_state["first_name"],
            "last_name": st.session_state["last_name"],
            "id_number": st.session_state["id_number"],
            "dob": st.session_state["dob"],
        })
        print("Extracted text...")

        with st.expander("Extracted text"):
            st.write(extracted_text['raw_text'])

        st.write(f"First name: {extracted_text['first_name']}")
        st.write(f"Last name: {extracted_text['last_name']}")
        st.write(f"Identification number: {extracted_text['id_number']}")
        st.write(f"Date of birth: {extracted_text['dob']}")

    except Exception as e:
        print("Could not show verification images: " + str(e))


def details_form():
    with st.sidebar.expander("Enter your details"):
        st.write("Your details will be compared to your legal document.")
        with st.form("compliance_form"):
            st.session_state["first_name"] = st.text_input(
                'First name', key="first_name")
            st.session_state["last_name"] = st.text_input(
                'Last name', key="last_name")
            st.session_state["id_number"] = st.text_input(
                'Identification number', key="id_number")
            st.session_state["dob"] = st.text_input(
                'Date of birth (as in document)', key="dob")
            st.form_submit_button(
                "Submit", use_container_width=True, type="primary")


def main():
    st.set_page_config(
        page_title="Smart Compliance", page_icon="✨")

    st.title("Smart Compliance ✨")
    st.write(
        "A fraud reduction strategy for financial institutions done in a few easy steps.")

    st.sidebar.title("Compliance documents")

    details_form()
    selfie_form()
    document_form()
    verification_form()
    info_form()

    st.sidebar.divider()
    st.sidebar.image("./hero.png", use_column_width=True,
                     caption="Smart Compliance - Le Wagon Project for Batch #1287 Mauritius with Clarice Bouwer, Amit Malik and Vighnesh Gaya.")


if __name__ == "__main__":
    print("Booting application...")
    main()
