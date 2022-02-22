from fastapi import FastAPI
import json
import uvicorn
from app.utils import Model

CONFIG_PATH = "app/Sidecards/config.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

model = Model(
    **config["model"]
                )

app = FastAPI(
    **config["api"]
)


@app.post("/predict")
async def get_template(text: str):
    return model.get_prediction(text)


#if __name__ == "__main__":
#    uvicorn.run("main:app")


