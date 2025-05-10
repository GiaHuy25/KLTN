from pydantic import BaseModel, Field
from typing import Optional, List
from decimal import Decimal
from datetime import datetime

# Disease Schemas
class DiseaseBase(BaseModel):
    name: str
    description: Optional[str] = None
    cause: Optional[str] = None
    symptoms: Optional[str] = None
    created_at: Optional[datetime] = None

class DiseaseCreate(DiseaseBase):
    pass

class DiseaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    cause: Optional[str] = None
    symptoms: Optional[str] = None

class Disease(DiseaseBase):
    id: int

    class Config:
        from_attributes = True
