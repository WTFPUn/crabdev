from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import cv2
import numpy as np
from Payloadtype import ImageDetection
from typing import List
from imagehandler import numpy_to_base64

CONFIDENCE_TRESHOLD = 0.3
CRAB_THRESHOLD = 1/8
MODEL_PATH = 'custom-model-3'

red_point = 242
green_point = 130
blue_point = 128

range_color = 50

red_range = (red_point - range_color, red_point + range_color)
green_range = (green_point - range_color, green_point + range_color)
blue_range = (blue_point - range_color, blue_point + range_color)


image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(device)

def get_box(path: str):
  image = cv2.imread(path)
  crop_image = []
  
  with torch.no_grad():
    # load image and predict
    inputs = image_processor(images=image, return_tensors='pt').to(device)
    outputs = model(**inputs)

    # post-process
    target_sizes = torch.tensor([image.shape[:2]]).to(device)
    results = image_processor.post_process_object_detection(
        outputs=outputs, 
        threshold=CONFIDENCE_TRESHOLD, 
        target_sizes=target_sizes
    )[0]

    # print(results)

    # crop class 2 only
    for score, labels, bbox in zip(results['scores'], results['labels'], results['boxes']):
        if labels == 1 :
            # topleft, bottomright
            x1, y1, x2, y2 = bbox
            crop = image[int(y1):int(y2), int(x1):int(x2)]

            # swarp bgr to rgb
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop_image.append(crop)
            
  return crop_image

def IsMolted(image: np.ndarray) -> bool:

  # from gb to rgb
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  N = image.shape[0] * image.shape[1]

  result = np.zeros_like(image)

  mask = (image[:, :, 0] > red_range[0]) & (image[:, :, 0] < red_range[1]) & \
          (image[:, :, 1] > green_range[0]) & (image[:, :, 1] < green_range[1]) & \
          (image[:, :, 2] > blue_range[0]) & (image[:, :, 2] < blue_range[1])
  
  result[mask] = image[mask]

  # dilate
  kernel = np.ones((5,5), np.uint8)
  result = cv2.dilate(result, kernel, iterations=1)


  # count non zero is more than 3/8 of total pixel
  if np.count_nonzero(result) > N * CRAB_THRESHOLD:
    return True
  
  return False

def detect_molted(path:str) -> List[ImageDetection]:
  crops = get_box(path)
  result: List[ImageDetection] = []
  for crop in crops:
    ismolt = IsMolted(crop)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    result.append(ImageDetection(image=numpy_to_base64(crop), molt=ismolt))
  
  return result