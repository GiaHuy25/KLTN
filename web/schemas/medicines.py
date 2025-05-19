from pydantic import BaseModel, Field
from typing import Optional, List
from decimal import Decimal
from datetime import datetime

class MedicineBase(BaseModel):
    medicine_id: int = Field(..., description="ID của thuốc")
    name: str = Field(..., min_length=1, max_length=100, description="Tên thuốc")
    description: Optional[str] = Field(None, max_length=1000, description="Mô tả về thuốc")
    price: Decimal = Field(..., gt=0, decimal_places=2, description="Giá của thuốc")
    stock: int = Field(0, ge=0, description="Số lượng tồn kho")
    image_url: Optional[str] = Field(None, max_length=255, description="URL của hình ảnh thuốc")
    image_base64: Optional[str] = Field(None, description="Hình ảnh thuốc dưới dạng Base64")
    how_to_use: Optional[str] = Field(None, max_length=1000, description="Hướng dẫn sử dụng")

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class MedicineCreate(MedicineBase):
    disease_id: int = Field(..., ge=1, description="ID của bệnh liên quan đến thuốc")

class MedicineUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Tên thuốc")
    description: Optional[str] = Field(None, max_length=1000, description="Mô tả về thuốc")
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2, description="Giá của thuốc")
    stock: Optional[int] = Field(None, ge=0, description="Số lượng tồn kho")
    disease_id: Optional[int] = Field(None, ge=1, description="ID của bệnh liên quan đến thuốc")
    image_url: Optional[str] = Field(None, max_length=255, description="URL của hình ảnh thuốc")
    how_to_use: Optional[str] = Field(None, max_length=1000, description="Hướng dẫn sử dụng")

class Medicine(MedicineBase):
    medicine_id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class PaginatedMedicineResponse(BaseModel):
    data: List[Medicine]
    total_records: int
    page: int
    per_page: int