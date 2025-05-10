from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi import HTTPException
from models.models import OrderDetails, Medicines
from schemas.order_details import OrderDetailCreate, OrderDetailUpdate


def create_order_detail(db: Session, order_detail_data: OrderDetailCreate) -> OrderDetails:
    try:
        medicine = db.query(Medicines).filter(Medicines.id == order_detail_data.medicine_id).first()
        if not medicine:
            raise HTTPException(status_code=404, detail="Medicine not found")
        
        if order_detail_data.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be greater than 0")
        
        db_order_detail = OrderDetails(
            order_id=order_detail_data.order_id,
            medicine_id=order_detail_data.medicine_id,
            quantity=order_detail_data.quantity,
            unit_price=order_detail_data.unit_price
        )
        db.add(db_order_detail)
        db.commit()
        db.refresh(db_order_detail)
        return db_order_detail
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating order detail: {str(e)}")


def get_order_details_by_order_id(db: Session, order_id: int) -> List[OrderDetails]:
    return db.query(OrderDetails).filter(OrderDetails.order_id == order_id).all()


def update_order_detail(db: Session, order_detail_id: int, order_detail_data: OrderDetailUpdate) -> OrderDetails:
    db_order_detail = db.query(OrderDetails).filter(OrderDetails.id == order_detail_id).first()
    if not db_order_detail:
        raise HTTPException(status_code=404, detail="Order detail not found")
    
    if order_detail_data.quantity is not None:
        if order_detail_data.quantity <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be greater than 0")
        db_order_detail.quantity = order_detail_data.quantity
    
    if order_detail_data.unit_price is not None:
        db_order_detail.unit_price = order_detail_data.unit_price
    
    db.commit()
    db.refresh(db_order_detail)
    return db_order_detail


def delete_order_detail(db: Session, order_detail_id: int) -> None:
    db_order_detail = db.query(OrderDetails).filter(OrderDetails.id == order_detail_id).first()
    if not db_order_detail:
        raise HTTPException(status_code=404, detail="Order detail not found")
    
    db.delete(db_order_detail)
    db.commit()
