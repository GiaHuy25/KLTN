from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from schemas.cart_item import CartItem  # Đảm bảo import đúng

class CartBase(BaseModel):
    user_id: int
    created_at: Optional[datetime] = None

class CartCreate(CartBase):
    pass

class Cart(CartBase):
    id: int

    class Config:
        from_attributes = True
