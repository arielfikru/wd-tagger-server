import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List

import flax
import jax
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import uvicorn
import base64

# You'll need to import your Models module or include the necessary model definitions here
import Models

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2_v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "swinv2_v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "vit-large": "SmilingWolf/wd-vit-large-tagger-v3",
}

@flax.struct.dataclass
class PredModel:
    apply_fun: Callable = flax.struct.field(pytree_node=False)
    params: Any = flax.struct.field(pytree_node=True)

    def jit_predict(self, x):
        x = x / 127.5 - 1
        x = self.apply_fun(self.params, x, train=False)
        x = flax.linen.sigmoid(x)
        x = jax.numpy.float32(x)
        return x

    def predict(self, x):
        preds = self.jit_predict(x)
        preds = jax.device_get(preds)
        preds = preds[0]
        return preds

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

def pil_resize(image: Image.Image, target_size: int) -> Image.Image:
    max_dim = max(image.size)
    if max_dim != target_size:
        image = image.resize((target_size, target_size), Image.BICUBIC)
    return image

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

def load_labels_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data

def load_model_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> PredModel:
    weights_path = hf_hub_download(repo_id=repo_id, filename="model.msgpack", revision=revision, token=token)
    model_config = hf_hub_download(repo_id=repo_id, filename="sw_jax_cv_config.json", revision=revision, token=token)

    with open(weights_path, "rb") as f:
        data = f.read()

    restored = flax.serialization.msgpack_restore(data)["model"]
    variables = {"params": restored["params"], **restored["constants"]}

    with open(model_config) as f:
        model_config = json.loads(f.read())

    model_name = model_config["model_name"]
    model_builder = Models.model_registry[model_name]()
    model = model_builder.build(config=model_builder, **model_config["model_args"])
    model = PredModel(model.apply, params=variables)
    return model, model_config["image_size"]

def get_tags(probs: Any, labels: LabelData, gen_threshold: float, char_threshold: float):
    probs = list(zip(labels.names, probs))
    rating_labels = dict([probs[i] for i in labels.rating])
    
    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    return rating_labels, char_labels, gen_labels

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded model and labels
model = None
labels = None
target_size = None

class TaggingResult(BaseModel):
    rating: str
    character_tags: List[str]
    general_tags: List[str]

class ImageRequest(BaseModel):
    image_base64: str
    gen_threshold: float = 0.35
    char_threshold: float = 0.75

@app.on_event("startup")
async def startup_event():
    global model, labels, target_size
    repo_id = MODEL_REPO_MAP["swinv2_v3"]  # Default to 'swinv2_v3' model
    model, target_size = load_model_hf(repo_id=repo_id)
    labels = load_labels_hf(repo_id=repo_id)

def process_image(img: Image.Image, gen_threshold: float, char_threshold: float) -> TaggingResult:
    img = pil_ensure_rgb(img)
    img = pil_pad_square(img)
    img = pil_resize(img, target_size)
    
    inputs = np.array(img)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = inputs[..., ::-1]

    outputs = model.predict(inputs)
    rating_labels, char_labels, gen_labels = get_tags(outputs, labels, gen_threshold, char_threshold)

    return TaggingResult(
        rating=max(rating_labels, key=rating_labels.get),
        character_tags=list(char_labels.keys()),
        general_tags=list(gen_labels.keys())
    )

@app.post("/tag_image/", response_model=TaggingResult)
async def tag_image(request: ImageRequest):
    try:
        # Remove the "data:image/jpeg;base64," part if it exists
        base64_data = request.image_base64.split(",")[-1]
        
        # Decode base64 string to bytes
        image_data = base64.b64decode(base64_data)
        
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    result = process_image(img, request.gen_threshold, request.char_threshold)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)