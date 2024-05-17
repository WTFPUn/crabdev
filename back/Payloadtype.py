from pydantic import BaseModel
from typing import Literal, Generic, TypeVar, TypedDict, List, Any

PayloadDtype = TypeVar('PayloadDtype', bound=Any)

class Payload(BaseModel, Generic[PayloadDtype]):
  type: str
  data: PayloadDtype 

class Image(BaseModel):
  image: str
  width: int
  height: int
  format: str

class ImagePayload(Payload[Image]):
  type: Literal['image'] = "image"
  
class Log(BaseModel):
  message: str
  level: str
  
class LogPayload(Payload[Log]):
  type: Literal['log'] = "log"
  
class ImageDetection(BaseModel):
  image: str
  format: str = "jpeg"
  molt: bool
  
class ImageCollection(BaseModel):
  images: list[ImageDetection]
  
class ImageCollectionPayload(Payload[List[ImageDetection]]):
  type: Literal['image_collection'] = "image_collection"

    