from fastapi import WebSocket
import numpy as np
import base64
import json
from Payloadtype import ImagePayload, ImageDetection
import cv2

async def send_image(websocket: WebSocket, image: np.ndarray):
  byte_image = cv2.imencode(".jpg", image)[1].tobytes()

  b64_image = base64.b64encode(byte_image).decode("utf-8").replace("\n", "")

  data = {
    "image": b64_image,
    "width": image.shape[1],
    "height": image.shape[0],
    "format": "jpeg"
  }
  
  image_payload = ImagePayload(data=data)
  # print(image_payload)

  await websocket.send_json(image_payload.model_dump_json())
  
def numpy_to_base64(image: np.ndarray) -> str:
  byte_image = cv2.imencode(".jpg", image)[1].tobytes()

  b64_image = base64.b64encode(byte_image).decode("utf-8").replace("\n", "")

  
  return b64_image
