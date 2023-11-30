- [x] Discard more than two faces on detection (Take the first two largest faces into account only)
- [x] Optimize detection of faces
- [±] Improve detection/verification speed
      - Removed additional similarity metrics
      - Remove OpenFace*
      - Only detected 2 faces so that the iteration is faster
- [x] Base image in legal document must be the same as the image in the selfie
- [x] Fix error handling with image face detection

- [ ] Work with rotated document
- [ ] [!!] Event loop is closed. App hangs and does not recover

- [ ] Harlem shake verification results grid (why is it shaking?)
- [ ] There seems to be an issue uploading different image types
- [ ] Check the exception printed on streamlit server
- [ ] Improve verification accuracy
- [ ] Adjust the UI to be more user friendly and not have all the options on the sidebar
- [ ] [!!!!!!!!] Create an algorithm to "predict" the outcome (green tick, red cross, yellow question mark)
      ±If 60% of total number of verified results then it is verified
- [ ] [!!!!!!!!] OCR for different types of documents: only cater for specific documents


Have a strong point and weak point of the model for the demo.
Majority vote for prediction.