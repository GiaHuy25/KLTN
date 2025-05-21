from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from decimal import Decimal

class OrderItem(BaseModel):
    medicine_id: int
    quantity: int
    price: Decimal = Field(..., gt=0, decimal_places=2)

# Schema tạo đơn hàng
class OrderCreate(BaseModel):
    user_id: int
    items: List[OrderItem]
    shipping_address: Optional[str] = None
    delivery_date: Optional[datetime] = None
    total_price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)

# Schema yêu cầu khi Checkout
class CheckoutRequest(BaseModel):
    user_id: int
    cart_item_ids: List[int]
    shipping_address: Optional[str] = None
    delivery_date: Optional[datetime] = None

# Schema trả về đơn hàng
class Order(BaseModel):
    order_id: int
    user_id: int
    total_price: Decimal
    status: str
    shipping_address: Optional[str] = None
    delivery_date: Optional[datetime] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
class OrderUpdate(BaseModel):
    user_id: Optional[int] = Field(None, description="ID của người dùng")
    status: Optional[str] = Field(None, description="Trạng thái của đơn hàng")
    shipping_address: Optional[str] = Field(None, description="Địa chỉ giao hàng")
    delivery_date: Optional[datetime] = Field(None, description="Ngày giao hàng dự kiến")
    total_price: Optional[Decimal] = Field(None, gt=0, decimal_places=2, description="Tổng giá trị đơn hàng")
