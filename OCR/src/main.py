from fastapi import FastAPI, Response

from src.routers.v1 import router as v1_router
from src.settings import settings

app = FastAPI(title='OCR Service',
              root_path=settings.ROOT_PATH, )


@app.get("/health", include_in_schema=False)
@app.get("/ping", include_in_schema=False)
async def health_check():
    return Response(status_code=200)


app.include_router(v1_router)
