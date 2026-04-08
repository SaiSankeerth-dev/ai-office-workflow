from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    try:
        return {
            "status": "running",
            "app": "AI Office Workflow Simulator",
            "message": "Deployment successful"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}
