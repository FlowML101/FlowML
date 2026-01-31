"""
API Routers package
"""
from routers.datasets import router as datasets_router
from routers.training import router as training_router
from routers.results import router as results_router
from routers.workers import router as workers_router
from routers.stats import router as stats_router
from routers.llm import router as llm_router

__all__ = [
    "datasets_router",
    "training_router",
    "results_router",
    "workers_router",
    "stats_router",
    "llm_router",
]
