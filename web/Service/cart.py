from sqlalchemy.orm import Session
from fastapi import HTTPException
from models.models import Cart


def get_cart_by_user_id(db: Session, user_id: int) -> Cart:
    db_cart = db.query(Cart).filter(Cart.user_id == user_id).first()
    if not db_cart:
        db_cart = create_cart(db, user_id)
    return db_cart


def create_cart(db: Session, user_id: int) -> Cart:
    db_cart = Cart(user_id=user_id)
    db.add(db_cart)
    db.commit()
    db.refresh(db_cart)
    return db_cart

