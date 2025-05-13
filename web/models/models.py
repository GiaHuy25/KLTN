from sqlalchemy import DateTime, Column, Text, DECIMAL, func, ForeignKey
from sqlalchemy.types import Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from decimal import Decimal

Base = declarative_base()

# User Model
class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    full_name = Column(String(100), nullable=True)
    phone_number = Column(String(15), nullable=True)
    address = Column(Text, nullable=True)
    role = Column(String(20), default='user')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Thêm mối quan hệ ngược lại
    carts = relationship("Cart", back_populates="user")
    orders = relationship("Orders", back_populates="user")

# Disease Model
class Disease(Base):
    __tablename__ = 'diseases'
    disease_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    cause = Column(Text, nullable=True)
    symptoms = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Thêm mối quan hệ với DiseaseMedicine
    disease_medicines = relationship("DiseaseMedicine", back_populates="disease")

# Medicine Model
class Medicines(Base):
    __tablename__ = 'medicines'
    medicine_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    how_to_use = Column(Text, nullable=True)
    price = Column(DECIMAL(10, 2), nullable=False)
    stock = Column(Integer, default=0)
    image_url = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Thêm mối quan hệ ngược lại
    cart_items = relationship("CartItems", back_populates="medicine")
    order_details = relationship("OrderItems", back_populates="medicine")
    disease_medicines = relationship("DiseaseMedicine", back_populates="medicine")

# Disease-Medicine Association Model
class DiseaseMedicine(Base):
    __tablename__ = 'disease_medicines'
    id = Column(Integer, primary_key=True, index=True)
    disease_id = Column(Integer, ForeignKey('diseases.disease_id'), nullable=False)
    medicine_id = Column(Integer, ForeignKey('medicines.medicine_id'), nullable=False)

    # Thêm mối quan hệ ngược lại
    disease = relationship("Disease", back_populates="disease_medicines")
    medicine = relationship("Medicines", back_populates="disease_medicines")

# Cart Model
class Cart(Base):
    __tablename__ = "cart"

    cart_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="carts")
    cart_items = relationship("CartItems", back_populates="cart")

# Cart Item Model
class CartItems(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cart_id = Column(Integer, ForeignKey("cart.cart_id"), nullable=False)
    medicine_id = Column(Integer, ForeignKey("medicines.medicine_id"), nullable=False)
    quantity = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())

    cart = relationship("Cart", back_populates="cart_items")
    medicine = relationship("Medicines", back_populates="cart_items")

# Order Model
class Orders(Base):
    __tablename__ = "orders"

    order_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    total_price = Column(DECIMAL(10, 2), nullable=False)
    status = Column(String(20), nullable=False, default="pending")
    shipping_address = Column(String, nullable=True)
    delivery_date = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="orders")
    order_details = relationship("OrderItems", back_populates="order")

# Order Item Model
class OrderItems(Base):
    __tablename__ = "order_items"

    item_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.order_id"), nullable=False)
    medicine_id = Column(Integer, ForeignKey("medicines.medicine_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(DECIMAL(10, 2), nullable=False)

    order = relationship("Orders", back_populates="order_details")
    medicine = relationship("Medicines", back_populates="order_details")