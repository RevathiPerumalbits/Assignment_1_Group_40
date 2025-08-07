from pydantic import BaseModel, conlist, PositiveFloat

class IrisFeatures(BaseModel):
    features: conlist(PositiveFloat, min_length=4, max_length=4)