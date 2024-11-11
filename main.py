from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="FastAPI Template",
    description="YOLOv8",
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


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")



# добавить метод получения метадаты