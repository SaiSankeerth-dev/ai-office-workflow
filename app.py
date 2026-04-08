from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AI Office Workflow Simulator is LIVE"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
