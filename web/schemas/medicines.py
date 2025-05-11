from pydantic import BaseModel, Field
from typing import Optional, List
from decimal import Decimal
from datetime import datetime

class MedicineBase(BaseModel):
    medicine_id: int = Field(..., description="ID của thuốc")
    name: str = Field(..., min_length=1, max_length=100, description="Tên thuốc")
    description: Optional[str] = Field(None, description="Mô tả về thuốc")
    price: Decimal = Field(..., gt=0, decimal_places=2, description="Giá của thuốc")
    discounted_price: Optional[Decimal] = Field(None, gt=0, decimal_places=2, description="Giá đã giảm (nếu có)")
    stock: int = Field(0, ge=0, description="Số lượng tồn kho")
    disease_id: Optional[int] = Field(None, description="ID của bệnh liên quan đến thuốc")
    image_url: Optional[str] = Field(None, description="URL của hình ảnh thuốc")
    image_base64: Optional[str] = Field(None, description="Hình ảnh thuốc dưới dạng Base64")
    is_freeship: bool = Field(False, description="Thuốc có miễn phí vận chuyển hay không")
    how_to_use: Optional[str] = Field(None, description="Hướng dẫn sử dụng")

# Schema để tạo mới thuốc
class MedicineCreate(MedicineBase):
    disease_id: int = Field(..., description="ID của bệnh liên quan đến thuốc")

# Schema để cập nhật thông tin thuốc
class MedicineUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    discounted_price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    stock: Optional[int] = Field(None, ge=0)
    disease_id: Optional[int] = Field(None, description="ID của bệnh liên quan đến thuốc")
    image_url: Optional[str] = None
    is_freeship: Optional[bool] = None
    how_to_use: Optional[str] = None

# Schema đại diện cho thuốc trong cơ sở dữ liệu
class Medicine(MedicineBase):
    medicine_id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PaginatedMedicineResponse(BaseModel):
    data: List[MedicineBase]
    total_records: int
    page: int
    per_page: int