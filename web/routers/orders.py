from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.db import get_db
from models.models import User as UserModel,  Medicines as MedicineModel
from schemas.orders import Order, OrderCreate, OrderUpdate
from schemas.order_details import OrderDetail
from schemas.medicines import Medicine as MedicineSchema
from Service import orders as order_controller
from Service import order_details as order_detail_controller
from Service import medicines as medicine_controller

router = APIRouter(
    prefix="/orders",
    tags=["Orders"],
)

# Sử dụng HTTPBearer cho xác thực token
security = HTTPBearer()

# Cấu hình JWT
SECRET_KEY = "your-secret-key"  # Thay bằng secret key thực tế, phải khớp với user_router.py
ALGORITHM = "HS256"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> UserModel:
    """
    Xác thực token và lấy thông tin user hiện tại.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user is None:
        raise credentials_exception
    return user

@router.get("/", response_model=List[Dict[str, Any]], summary="Get user's orders")
async def get_orders(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Lấy danh sách đơn hàng của người dùng.
    """
    try:
        orders = order_controller.get_orders_by_user_id(db, user_id=current_user.user_id)
        result = []
        for order in orders:
            order_details = order_detail_controller.get_order_details_by_order_id(db, order_id=order.order_id)
            details_with_medicines = []
            for detail in order_details:
                medicine = medicine_controller.get_medicine_by_id(db, medicine_id=detail.medicine_id)
                if medicine:
                    details_with_medicines.append({
                        "detail_id": detail.item_id,
                        "quantity": detail.quantity,
                        "unit_price": float(detail.price),
                        "medicine": MedicineSchema.from_orm(medicine)
                    })
            result.append({
                "order_id": order.order_id,
                "user_id": order.user_id,
                "created_at": order.created_at,
                "total_price": float(order.total_price),
                "status": order.status,
                "Shipping address": order.shipping_address,
                "details": details_with_medicines
            })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving orders: {str(e)}")

@router.get("/{order_id}", response_model=Dict[str, Any], summary="Get order details by ID")
async def get_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Lấy chi tiết đơn hàng theo ID.
    """
    order = order_controller.get_order_by_id(db, order_id=order_id)
    if not order or order.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Order not found or not authorized")

    order_details = order_detail_controller.get_order_details_by_order_id(db, order_id=order_id)
    details_with_medicines = []
    for detail in order_details:
        medicine = medicine_controller.get_medicine_by_id(db, medicine_id=detail.medicine_id)
        if medicine:
            details_with_medicines.append({
                "detail_id": detail.item_id,
                "quantity": detail.quantity,
                "unit_price": float(detail.price),
                "medicine": MedicineSchema.from_orm(medicine)
            })

    return {
        "order_id": order.order_id,
        "user_id": order.user_id,
        "created_at": order.created_at,
        "total_price": float(order.total_price),
        "status": order.status,
        "Shipping address": order.shipping_address,
        "details": details_with_medicines
    }

@router.post("/", response_model=Order, summary="Create a new order")
async def create_order(
    order_data: OrderCreate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    if current_user.user_id != order_data.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to create order for this user")
    try:
        new_order = order_controller.create_order(db, order_data=order_data)
        return Order.from_orm(new_order)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating order: {str(e)}")

@router.put("/{order_id}", response_model=Order, summary="Update order")
async def update_order(
    order_id: int,
    order_data: OrderUpdate,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Cập nhật thông tin đơn hàng.
    """
    order = order_controller.get_order_by_id(db, order_id=order_id)
    if not order or order.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Order not found or not authorized")

    try:
        updated_order = order_controller.update_order(db, order_id=order_id, order_data=order_data)
        return updated_order
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating order: {str(e)}")

@router.delete("/{order_id}", status_code=status.HTTP_200_OK, summary="Delete order")
async def delete_order(
    order_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Xóa đơn hàng theo ID.
    """
    order = order_controller.get_order_by_id(db, order_id=order_id)
    if not order or order.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Order not found or not authorized")

    try:
        order_controller.delete_order(db, order_id=order_id)
        return {"message": "Order deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting order: {str(e)}")