from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException
from models.models import Orders, Medicines, OrderItems
from schemas.orders import OrderCreate, OrderUpdate
from datetime import datetime
from decimal import Decimal

def create_order(db: Session, order_data: OrderCreate) -> Orders:
    try:
        # Kiểm tra danh sách items
        if not order_data.items:
            raise HTTPException(status_code=400, detail="Order items cannot be empty")

        # Tính tổng giá trị đơn hàng và kiểm tra tồn kho
        total_price = Decimal('0')
        for item in order_data.items:
            # Kiểm tra số lượng tồn kho
            medicine = db.query(Medicines).filter(Medicines.medicine_id == item.medicine_id).first()
            if not medicine:
                raise HTTPException(status_code=404, detail=f"Medicine {item.medicine_id} not found")
            if medicine.stock < item.quantity:
                raise HTTPException(status_code=400, detail=f"Insufficient stock for item {medicine.name}")

            # Giảm số lượng tồn kho
            medicine.stock -= item.quantity
            db.add(medicine)

            # Tính tổng tiền
            total_price += item.price * Decimal(str(item.quantity))

        # Tạo đơn hàng mới
        db_order = Orders(
            user_id=order_data.user_id,
            total_price=float(total_price),
            status="pending",
            created_at=datetime.now(),
            shipping_address=order_data.shipping_address,
            delivery_date=order_data.delivery_date
        )
        db.add(db_order)
        db.commit()
        db.refresh(db_order)

        # Tạo chi tiết đơn hàng
        for item in order_data.items:
            db_order_item = OrderItems(
                order_id=db_order.order_id,
                medicine_id=item.medicine_id,
                quantity=item.quantity,
                price=float(item.price)
            )
            db.add(db_order_item)

        db.commit()
        return db_order

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating order: {str(e)}")

def get_order_by_id(db: Session, order_id: int) -> Orders:
    db_order = db.query(Orders).filter(Orders.order_id == order_id).first()
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
    db_order = db.query(Orders).filter(Orders.order_id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")

    db.query(OrderItems).filter(OrderItems.order_id == order_id).delete()
    db.delete(db_order)
    db.commit()
