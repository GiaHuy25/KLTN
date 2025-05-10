from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class MedicineCategoryBase(BaseModel):
    name: str
    description: Optional[str] = None

class MedicineCategoryCreate(MedicineCategoryBase):
    pass

class MedicineCategoryUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class MedicineCategory(MedicineCategoryBase):
    id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True