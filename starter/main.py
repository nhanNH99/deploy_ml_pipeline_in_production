"""
Author: Tran Duy Nhat Anh
Project: Udacity - MLOPS Nanodegree - Project 3
Date: 21 - 08 - 2024
"""

import json
import logging
import os
from dataclasses import dataclass, field

from fastapi import FastAPI, Request

from starter.predict_model import main_inference
from starter.src.helper_utils import DataInput


@dataclass
class ai_serving:
    app: "FastAPI" = field(default_factory=FastAPI)

    def __post_init__(self):
        self.app.get("/")(self.get_info)
        self.app.post("/inference")(self.inference)

    @staticmethod
    async def get_info():
        return {"message": "Welcome to the MLOps project of Tran Duy Nhat Anh"}

    async def inference(self, request: Request, input_data: DataInput):
        try:
            body = await request.json()
            result = main_inference(DataInput(**body).dict())
            return {"message": str(result), "status": 200}
        except ImportError as e:
            return {"message": str(e), "status": 422}


ai = ai_serving().app
