"""A3 HTTP adapter using the existing host Python model runtime."""
from pathlib import Path
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from .inference_views import register_contract_rag

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
from ai_extract.services.inference_runtime_service import configure_gateway_logging
configure_gateway_logging()
if os.getenv('BGE_HOST_CACHE'):
    os.environ.setdefault('BGE_LOCAL_CACHE', os.environ['BGE_HOST_CACHE'])
app = FastAPI(title='Local A3 inference')
register_contract_rag(app)
