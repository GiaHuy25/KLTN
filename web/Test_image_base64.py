import base64
from io import BytesIO
from PIL import Image

# Chuỗi base64 của hình ảnh (thay thế bằng chuỗi của bạn)
image_base64 = "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"

# Hàm để hiển thị hình ảnh từ chuỗi base64
def display_image(base64_string):
    try:
        # Giải mã chuỗi base64
        image_data = base64.b64decode(base64_string)
        
        # Chuyển dữ liệu thành đối tượng BytesIO để PIL xử lý
        image_stream = BytesIO(image_data)
        
        # Mở hình ảnh bằng PIL
        image = Image.open(image_stream)
        
        # Hiển thị hình ảnh
        image.show()
        
        return "Hình ảnh đã được hiển thị thành công!"
    except Exception as e:
        return f"Error: {str(e)}"

# Thực hiện hiển thị hình ảnh
result = display_image(image_base64)
print(result)