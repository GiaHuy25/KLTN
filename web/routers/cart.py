from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from config.db import get_db
from models.models import User as UserModel, Cart as CartModel, CartItems as CartItemModel, Medicines as MedicineModel
from schemas.cart import Cart
from schemas.cart_item import CartItem, CartItemCreate, CartItemUpdate, CartItemRequest, CartItemUpdateRequest
from schemas.medicines import Medicine as MedicineSchema
from schemas.orders import Order
from Service import cart as cart_controller, cart_item as cart_items_controller
from Service import orders as order_controller

router = APIRouter(
    prefix="/cart",
    tags=["Carts"],
)

def get_or_create_user_cart(db: Session, user_id: int) -> CartModel:
    """
    Lấy hoặc tạo giỏ hàng chưa thanh toán cho người dùng.
    """
    db_cart = cart_controller.get_cart_by_user_id(db, user_id=user_id)
    if not db_cart:
        db_cart = cart_controller.create_cart(db, user_id=user_id)
    return db_cart

@router.post("/items", response_model=CartItem, status_code=status.HTTP_201_CREATED, summary="Add a medicine to the cart")
async def add_item(
    item_data: CartItemRequest,
    db: Session = Depends(get_db)
):
    """
    Thêm một thuốc vào giỏ hàng của người dùng.
    Nếu thuốc đã có, số lượng sẽ được cộng dồn.
    """
    user = db.query(UserModel).filter(UserModel.user_id == item_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db_cart = get_or_create_user_cart(db, user_id=item_data.user_id)

    item_to_add = CartItemCreate(
        product_id=item_data.product_id,
        quantity=item_data.quantity
    )
    return cart_items_controller.add_item_to_cart(db, cart_id=db_cart.cart_id, item_data=item_to_add)

@router.get("/", response_model=Dict[str, Any], summary="Get the user's cart details")
async def get_cart(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Lấy thông tin chi tiết giỏ hàng của người dùng, bao gồm danh sách thuốc.
    """
    user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db_cart = get_or_create_user_cart(db, user_id=user_id)
    cart_items = cart_items_controller.get_items_in_cart(db, cart_id=db_cart.cart_id)

    medicine_ids = [item.medicine_id for item in cart_items]
    medicines = db.query(MedicineModel).filter(MedicineModel.medicine_id.in_(medicine_ids)).all()
    medicine_dict = {m.medicine_id: m for m in medicines}

    items_with_details = []
    total_price = 0
    for item in cart_items:
        medicine = medicine_dict.get(item.medicine_id)
        if medicine:
            subtotal = float(medicine.price) * item.quantity
            item_detail = {
                "item_id": item.id,
                "quantity": item.quantity,
                "medicine": MedicineSchema.from_orm(medicine),
                "subtotal": subtotal
            }
            items_with_details.append(item_detail)
            total_price += subtotal

    return {
        "cart_id": db_cart.cart_id,
        "user_id": db_cart.user_id,
        "items": items_with_details,
        "total_price": total_price
    }

@router.put("/items/{item_id}", response_model=CartItem, summary="Update item quantity in the cart")
async def update_item_quantity(
    item_id: int,
    item_data: CartItemUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Cập nhật số lượng của một thuốc trong giỏ hàng.
    """
    db_item = db.query(CartItemModel).filter(CartItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")

    item_update = CartItemUpdate(quantity=item_data.quantity)
    return cart_items_controller.update_cart_item(db, item_id=item_id, item_data=item_update)

@router.delete("/items/{item_id}", status_code=status.HTTP_200_OK, summary="Remove a medicine from the cart")
async def remove_item(
    item_id: int,
    db: Session = Depends(get_db)
):
    """
    Xóa một thuốc khỏi giỏ hàng.
    """
    db_item = db.query(CartItemModel).filter(CartItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")

    cart_items_controller.delete_cart_item(db, item_id=item_id)
    return {"message": "Item removed successfully"}

@router.post("/checkout", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED, summary="Checkout and create an order")
async def checkout_cart(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Thanh toán giỏ hàng, tạo đơn hàng mới.
    """
    user = db.query(UserModel).filter(UserModel.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db_cart = get_or_create_user_cart(db, user_id=user_id)
    cart_items = cart_items_controller.get_items_in_cart(db, cart_id=db_cart.cart_id)
    if not cart_items:
        raise HTTPException(status_code=400, detail="Cart is empty")

    db_order = order_controller.create_order(db, user_id=user_id, cart_id=db_cart.cart_id)

    # Tạo giỏ hàng mới sau khi thanh toán
    new_cart = cart_controller.create_cart(db, user_id=user_id)

    return {
        "message": "Checkout successful",
        "order": Order.from_orm(db_order),
        "new_cart_id": new_cart.cart_id
    }
