from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CartItemBase(BaseModel):
    medicine_id: int
    quantity: int

class CartItemCreate(CartItemBase):
    pass

class CartItemUpdate(BaseModel):
    quantity: Optional[int] = None

class CartItemRequest(BaseModel):
    medicine_id: int
    quantity: int
    user_id: int

class CartItemUpdateRequest(BaseModel):
    quantity: int
    user_id: int

class CartItem(CartItemBase):
    medicine_id: int
    cart_id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
