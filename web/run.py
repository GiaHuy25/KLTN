import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(2)
    webbrowser.open("http://localhost:8000/docs")

if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=2,       
        log_level="info" 
    )
