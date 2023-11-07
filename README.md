# Smart Compliance

## UI component:
Ask for name, last name, id / passport / nic, dob + upload photo of face, holding id document. Create a Streamlit front end application.

## Jupyter notebook

### Jupyter notebook (part 1):
Detect the faces

https://www.kaggle.com/code/serkanpeldek/face-detection-with-opencv

### Jupyter notebook (part 2):
Compare two photos of people and make a match.

https://github.com/serengil/deepface
https://jonascleveland.com/best-algorithms-for-face-recognition/
https://www.youtube.com/watch?v=WnUVYQP4h44 (DeepFace: A Facial Recognition Library for Python)

### Jupyter notebook (part 3):
Extract text from id document (OCR)

https://huggingface.co/models?pipeline_tag=image-to-text

## API:
Get the photo, detect and compare photo of person with their ID card photo and verify the personal information provided that is read from the ID document.

## UI component:
Output the result to the UI to say the “fraud probability score / percentage”

## Deployment considerations:

ML Flow:
- Docker for API
- GCP for API
- Streamlit cloud for UI