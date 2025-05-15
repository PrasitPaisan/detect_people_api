from fastapi import FastAPI
from app.controllers import body_attributes_controller , detection_controler

app = FastAPI()

app.include_router(detection_controler.router, prefix="/api",)
app.include_router(body_attributes_controller.router, prefix="/api")