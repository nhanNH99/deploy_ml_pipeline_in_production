from injector import inject, singleton
from pydantic import BaseModel, Field


@singleton
class Settings:
    @inject
    def __init__(self, config: dict):
        pass


class DataInput(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlwgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 45,
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "education": "Bachelors",
                    "education_num": 13,
                    "fnlwgt": 2334,
                    "hours_per_week": 60,
                    "marital_status": "Never-married",
                    "native_country": "Cuba",
                    "occupation": "Prof-specialty",
                    "race": "Black",
                    "relationship": "Wife",
                    "sex": "Female",
                    "workclass": "State-gov",
                }
            ]
        }
    }