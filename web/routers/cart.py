from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config.db import get_db
from models.models import User as UserModel, Cart as CartModel, CartItems as CartItemModel, Medicines as MedicineModel, CartItems
from schemas.cart_item import CartItem, CartItemCreate, CartItemUpdate, CartItemRequest, CartItemUpdateRequest
from schemas.medicines import Medicine as MedicineSchema
from schemas.orders import OrderItem ,OrderCreate, CheckoutRequest
from Service import cart as cart_controller, cart_item as cart_items_controller
from Service import orders
from decimal import Decimal

router = APIRouter(
    prefix="/cart",
    tags=["Carts"],
)

security = HTTPBearer()
SECRET_KEY = "your-secret-key"  # Thay bằng secret key thực tế
ALGORITHM = "HS256"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
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

def get_or_create_user_cart(db: Session, user_id: int) -> CartModel:
    """
    Lấy hoặc tạo giỏ hàng chưa thanh toán cho người dùng.
    """
    db_cart = cart_controller.get_cart_by_user_id(db, user_id=user_id)
    if not db_cart:
        db_cart = cart_controller.create_cart(db, user_id=user_id)
        if not db_cart:  # Nếu không tạo được giỏ hàng
            raise HTTPException(status_code=500, detail="Failed to create cart for this user")
    return db_cart

@router.post("/items", response_model=CartItem, status_code=status.HTTP_201_CREATED, summary="Add a medicine to the cart")
async def add_item(
    item_data: CartItemRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    if current_user.user_id != item_data.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to add items to this user's cart")

    medicine = db.query(MedicineModel).filter(MedicineModel.medicine_id == item_data.medicine_id).first()
    if not medicine:
        raise HTTPException(status_code=404, detail="Medicine not found")

    db_cart = get_or_create_user_cart(db, user_id=item_data.user_id)
    item_to_add = CartItemCreate(medicine_id=item_data.medicine_id, quantity=item_data.quantity)
    return cart_items_controller.add_item_to_cart(db, cart_id=db_cart.cart_id, item_data=item_to_add)


@router.get("/", response_model=Dict[str, Any], summary="Get the user's cart details")
async def get_cart(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    if current_user.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this user's cart")

    db_cart = get_or_create_user_cart(db, user_id=user_id)
    cart_items = cart_items_controller.get_items_in_cart(db, cart_id=db_cart.cart_id)
    
    if not cart_items:
        return {
            "cart_id": db_cart.cart_id,
            "user_id": user_id,
            "items": [],
            "total_price": 0.0
        }

    # Lấy thông tin các loại thuốc từ danh sách item
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
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Cập nhật số lượng của một thuốc trong giỏ hàng.
    """
    db_item = db.query(CartItemModel).filter(CartItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")
    db_cart = db.query(CartModel).filter(CartModel.cart_id == db_item.cart_id).first()
    if db_cart.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this cart item")
    item_update = CartItemUpdate(quantity=item_data.quantity)
    return cart_items_controller.update_cart_item(db, item_id=item_id, item_data=item_update)

@router.delete("/items/{item_id}", status_code=status.HTTP_200_OK, summary="Remove a medicine from the cart")
async def remove_item(
    item_id: int,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    """
    Xóa một thuốc khỏi giỏ hàng.
    """
    db_item = db.query(CartItemModel).filter(CartItemModel.id == item_id).first()
    if not db_item:
        raise HTTPException(status_code=404, detail="Cart item not found")
    db_cart = db.query(CartModel).filter(CartModel.cart_id == db_item.cart_id).first()
    if db_cart.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Not authorized to remove this cart item")
    cart_items_controller.delete_cart_item(db, item_id=item_id)
    return {"message": "Item removed successfully"}

@router.post("/checkout", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED, summary="Checkout and create an order")
async def checkout_cart(
    checkout_data: CheckoutRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    try:
        # Kiểm tra quyền truy cập
        if checkout_data.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="You can only checkout your own cart")

        # Lấy giỏ hàng của người dùng
        user_cart = db.query(CartModel).filter(CartModel.user_id == checkout_data.user_id).first()
        if not user_cart:
            raise HTTPException(status_code=404, detail="Cart not found for this user")

        # Lấy các mục trong giỏ hàng dựa trên danh sách cart_item_ids
        cart_items = db.query(CartItems).filter(
            CartItems.cart_id == user_cart.cart_id,
            CartItems.id.in_(checkout_data.cart_item_ids)
        ).all()

        if not cart_items:
            raise HTTPException(status_code=400, detail="No items found in cart for the provided cart item IDs")

        # Kiểm tra số lượng cart_item_ids
        if len(cart_items) != len(checkout_data.cart_item_ids):
            raise HTTPException(status_code=400, detail="Some cart item IDs are invalid or do not belong to this cart")

        # Tạo danh sách OrderItem từ cart_items
        order_items = []
        for item in cart_items:
            # Lấy thông tin giá từ bảng medicines
            medicine = db.query(MedicineModel).filter(MedicineModel.medicine_id == item.medicine_id).first()
            if not medicine:
                raise HTTPException(status_code=404, detail=f"Medicine {item.medicine_id} not found")

            order_items.append(OrderItem(
                medicine_id=item.medicine_id,
                quantity=item.quantity,
                price=Decimal(str(medicine.price))
            ))

        # Tạo đối tượng OrderCreate
        order_create = OrderCreate(
            user_id=checkout_data.user_id,
            items=order_items,
            shipping_address=checkout_data.shipping_address,
            delivery_date=checkout_data.delivery_date
        )

        # Gọi service để tạo đơn hàng
        db_order = orders.create_order(db, order_data=order_create)

        # Xóa các mục đã đặt khỏi cart_items
        db.query(CartItems).filter(
            CartItems.id.in_(checkout_data.cart_item_ids)
        ).delete(synchronize_session=False)

        db.commit()

        return {
            "message": "Checkout successful",
            "order_id": db_order.order_id,
            "total_price": db_order.total_price,
            "status": db_order.status
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating order: {str(e)}")

