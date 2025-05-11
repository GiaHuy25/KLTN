from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.db import engine, Base
from routers import diseases, medicines, users, cart, orders

app = FastAPI()

# --- Cấu hình CORS ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "https://your-frontend-domain.com",
    "http://192.168.10.132:5173",
    "http://192.168.1.194:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Kết thúc cấu hình CORS ---

# Khởi tạo cơ sở dữ liệu
Base.metadata.create_all(bind=engine)

# Đăng ký router với prefix và tags
app.include_router(users.router, tags=["Users"])
app.include_router(medicines.router, tags=["Medicines"])
app.include_router(diseases.router, tags=["Diseases"])
app.include_router(cart.router, tags=["Carts"])
app.include_router(orders.router, tags=["Orders"])

# Route gốc để kiểm tra API
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "API for Durian Disease Management and Medicine Store is running!"}
