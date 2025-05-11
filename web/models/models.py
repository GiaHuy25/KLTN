from sqlalchemy import DateTime, Column, Text, DECIMAL, func, Boolean, ForeignKey
from sqlalchemy.types import Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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

# Disease Model
class Disease(Base):
    __tablename__ = 'diseases'
    disease_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    cause = Column(Text, nullable=True)
    symptoms = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


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

# Disease-Medicine Association Model
class DiseaseMedicine(Base):
    __tablename__ = 'disease_medicines'
    id = Column(Integer, primary_key=True, index=True)
    disease_id = Column(Integer, ForeignKey('diseases.disease_id'), nullable=False)
    medicine_id = Column(Integer, ForeignKey('medicines.medicine_id'), nullable=False)

# Cart Model
class Cart(Base):
    __tablename__ = 'cart'
    cart_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Cart Item Model
class CartItems(Base):
    __tablename__ = 'cart_items'
    id = Column(Integer, primary_key=True, index=True)
    cart_id = Column(Integer, ForeignKey('cart.cart_id'), nullable=False)
    medicine_id = Column(Integer, ForeignKey('medicines.medicine_id'), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Order Model
class Orders(Base):
    __tablename__ = 'orders'
    order_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), nullable=True)
    total_price = Column(DECIMAL(10, 2), nullable=False)
    status = Column(String(20), default='pending')
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Order Item Model
class OrderDetails(Base):
    __tablename__ = 'order_details'
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey('orders.order_id'), nullable=False)
    medicine_id = Column(Integer, ForeignKey('medicines.medicine_id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(DECIMAL(10, 2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
