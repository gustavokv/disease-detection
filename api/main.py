from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.image import resize

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model('../saved_models/efficientnetv2m_db2_480_corn_classifier.keras')
#CLASS_NAMES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
CLASS_NAMES = ["Gray Leaf Spot", "Northern Leaf Blight", "Northern Leaf Spot"]

def read_file_as_image(data) -> np.ndarray:
    img = np.array(Image.open(BytesIO(data)))
    return img

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = read_file_as_image(await file.read())
    img_resized = resize(img, [480,480])
    img_batch = np.expand_dims(img_resized, 0)
    
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    print(predicted_class)
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)