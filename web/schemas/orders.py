from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
from decimal import Decimal

class OrderBase(BaseModel):
    user_id: Optional[int] = None
    order_date: Optional[date] = None
    total_amount: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    status: str = Field(default="Pending")
    shipping_address: Optional[str] = None
    delivery_date: Optional[date] = None

class OrderCreate(OrderBase):
    pass

class OrderUpdate(BaseModel):
    user_id: Optional[int] = None
    status: Optional[str] = None
    shipping_address: Optional[str] = None
    delivery_date: Optional[date] = None

class Order(OrderBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            date: lambda v: v.isoformat() if v else None
        }
