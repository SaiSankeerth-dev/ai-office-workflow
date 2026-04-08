from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running", "message": "Server entry working"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return {"status": "reset successful"}