from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException
from models.models import CartItems, Medicines
from schemas.cart_item import CartItemUpdate, CartItemRequest

def add_item_to_cart(db: Session, cart_id: int, item_data: CartItemRequest):
    medicine = db.query(Medicines).filter(Medicines.medicine_id == item_data.medicine_id).first()
    if not medicine:
        raise HTTPException(status_code=404, detail="Medicine not found")

    # Tìm xem sản phẩm đã có trong giỏ hàng chưa
    cart_item = db.query(CartItems).filter(
        CartItems.cart_id == cart_id,
        CartItems.medicine_id == item_data.medicine_id
    ).first()

    # Nếu đã có, cập nhật số lượng
    if cart_item:
        cart_item.quantity = item_data.quantity
    else:
        # Nếu chưa có, tạo mới
        cart_item = CartItems(
            cart_id=cart_id,
            medicine_id=item_data.medicine_id,
            quantity=item_data.quantity
        )
        db.add(cart_item)

    db.commit()
    db.refresh(cart_item)
    return cart_item



def get_items_in_cart(db: Session, cart_id: int) -> List[CartItems]:
    return db.query(CartItems).filter(CartItems.cart_id == cart_id).all()


def update_cart_item(db: Session, item_id: int, item_data: CartItemUpdate) -> CartItems:
    db_item = db.query(CartItems).filter(CartItems.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")

    if item_data.quantity is not None:
        db_item.quantity = item_data.quantity
    db.commit()
    db.refresh(db_item)
    return db_item


def delete_cart_item(db: Session, item_id: int):
    db_item = db.query(CartItems).filter(CartItems.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")

    db.delete(db_item)
    db.commit()
