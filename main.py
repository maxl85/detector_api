from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="FastAPI Template",
    description="YOLOv8n",
    version="1.0",
)

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

class RequestModel(BaseModel):
    image: str  # base64 encoded image
    detector_name: str
    function_name: str


@app.get("/")
async def main():
    """ Entry point for the application """
    return {"message": "Welcome to FastAPI"}

