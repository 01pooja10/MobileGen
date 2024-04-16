from fastapi import FastAPI, Response, Form
from fastapi.responses import StreamingResponse, FileResponse
from typing import Optional
import tempfile
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from DDPM.scripts.updated_inference import Diffusion

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_image(txt: str, ver:int):
    
    classes = {'airplane':0, 'auto':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    label = classes[txt]
    num_imgs = 1
    mpath = '/content/ema_student.ckpt'
    
    diffusion = Diffusion()

    img_out,time = diffusion.infer(mpath,label,num_imgs,cfg=0)
    image_path = '/local/path'
    
    cv2.imwrite(image_path,img_out)
    
    print("Inference time:",time)
    
    return image_path

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate_image")
async def get_image(text: str = Form(...), version: int = Form(...)):
    image_path = Path(generate_image(text, version))
    if not image_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)
