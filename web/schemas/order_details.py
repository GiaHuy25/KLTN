from pydantic import BaseModel, Field
from typing import Optional
from decimal import Decimal
from datetime import datetime

class OrderDetailBase(BaseModel):
    order_id: Optional[int] = None
    medicine_id: Optional[int] = None
    quantity: int = Field(..., gt=0, description="Số lượng sản phẩm")
    unit_price: Decimal = Field(..., gt=0, decimal_places=2, description="Đơn giá sản phẩm")

class OrderDetailCreate(OrderDetailBase):
    pass

class OrderDetailUpdate(BaseModel):
    quantity: Optional[int] = Field(None, gt=0)
    unit_price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)

class OrderDetail(OrderDetailBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
