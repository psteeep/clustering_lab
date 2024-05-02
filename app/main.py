import uvicorn
from fastapi import FastAPI
from .routers.cl_endpoints import router as cl_router

app = FastAPI()
app.include_router(cl_router)


@app.get('/')
async def health_check():
    return {
        "status_code": 200,
        "details": "ok",
        "result": "working"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
