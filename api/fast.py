import os
from deepface import DeepFace
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(first_name: Annotated[str, Form()],
                  last_name: Annotated[str, Form()],
                  photo: Annotated[UploadFile, File()]):
    with open(f"./raw_data/{photo.filename}", "wb") as buffer:
        buffer.write(await photo.read())

    return {"first_name": first_name, "last_name": last_name, "photo": photo.filename}
    # result = DeepFace.verify(img1_path = "photo1.jpg", img2_path = "photo2.jpg", model_name = 'VGG-Face', distance_metric = 'euclidean_l2')
    # os.remove("photo1.jpg")
    # os.remove("photo2.jpg")
    # return result


@app.get("/")
def root():
    return {"message": "Hello Amit"}