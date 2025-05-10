from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException
from models.models import OrderDetails, Orders, CartItems, Medicines
from schemas.orders import OrderCreate, OrderUpdate

def create_order(db: Session, user_id: int, cart_id: int) -> Orders:
    try:
        cart_items = db.query(CartItems).filter(CartItems.cart_id == cart_id).all()
        if not cart_items:
            raise HTTPException(status_code=400, detail="Cart is empty")

        total_amount = 0
        for item in cart_items:
            medicine = db.query(Medicines).filter(Medicines.id == item.medicine_id).first()
            if not medicine:
                raise HTTPException(status_code=404, detail=f"Medicine {item.medicine_id} not found")
            total_amount += medicine.price * item.quantity

        db_order = Orders(
            user_id=user_id,
            total_amount=total_amount,
            status="Pending"
        )
        db.add(db_order)
        db.commit()
        db.refresh(db_order)

        for item in cart_items:
            medicine = db.query(Medicines).filter(Medicines.id == item.medicine_id).first()
            db_order_detail = OrderDetails(
                order_id=db_order.id,
                medicine_id=item.medicine_id,
                quantity=item.quantity,
                unit_price=medicine.price
            )
            db.add(db_order_detail)

        db.commit()
        return db_order
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating order: {str(e)}")

def get_order_by_id(db: Session, order_id: int) -> Orders:
    db_order = db.query(Orders).filter(Orders.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")
    return db_order

def get_orders_by_user_id(db: Session, user_id: int) -> List[Orders]:
    return db.query(Orders).filter(Orders.user_id == user_id).all()

def update_order(db: Session, order_id: int, order_data: OrderUpdate) -> Orders:
    db_order = db.query(Orders).filter(Orders.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order_data.user_id is not None:
        db_order.user_id = order_data.user_id
    if order_data.status is not None:
        db_order.status = order_data.status
    if order_data.shipping_address is not None:
        db_order.shipping_address = order_data.shipping_address
    if order_data.delivery_date is not None:
        db_order.delivery_date = order_data.delivery_date

    db.commit()
    db.refresh(db_order)
    return db_order

def delete_order(db: Session, order_id: int) -> None:
    db_order = db.query(Orders).filter(Orders.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")

    db.query(OrderDetails).filter(OrderDetails.order_id == order_id).delete()
    db.delete(db_order)
    db.commit()
