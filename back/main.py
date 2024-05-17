from fastapi import FastAPI, WebSocket
from imagehandler import send_image
import numpy as np
import cv2
import asyncio
import os
from Payloadtype import LogPayload, ImageCollectionPayload
from detector import detect_molted
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
    detected = detect_molted(image_test[idx])
    payload = ImageCollectionPayload(data=detected)
    print([d.molt for d in detected])
    await websocket.send_json(payload.model_dump_json())
    end = time()
    if end - start < polling_time:
      await asyncio.sleep(polling_time - (end - start))
    idx = (idx + 1) % len(image_test)

