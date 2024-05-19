from fastapi import FastAPI, WebSocket, Request, Response, UploadFile, File
from imagehandler import send_image
import numpy as np
import cv2
import asyncio
import os
from Payloadtype import LogPayload, ImageCollectionPayload
from detector import detect_molted, load_image
from time import time


polling_time = 5


app = FastAPI()


def simulate_image_shifting():
  folder = "images_test/"
  image_names = os.listdir(folder)
  
  return [folder + image for image in image_names]
  

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  
  # test molted detection
  image_test = simulate_image_shifting()
  # await send_image(websocket, image_test)
  idx = 0
  while True:
    start = time()
    shift_image = load_image(image_test[idx])
    detected = detect_molted(shift_image)
    payload = ImageCollectionPayload(data=detected)
    print([d.molt for d in detected])
    await websocket.send_json(payload.model_dump_json())
    end = time()
    if end - start < polling_time:
      await asyncio.sleep(polling_time - (end - start))
    idx = (idx + 1) % len(image_test)

# post method to receive image
@app.post("/predict")
async def image_endpoint(file: UploadFile = File(...)):
  image = await file.read()
  image = np.frombuffer(image, np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  detected = detect_molted(image)
  print(len(detected))
  payload = ImageCollectionPayload(data=detected)
  return payload.model_dump_json()