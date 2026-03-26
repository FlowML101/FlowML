import asyncio
from backend.database import get_session_context
from backend.models.job import Job
from sqlmodel import select
async def main():
    async with get_session_context() as session:
        result = await session.execute(select(Job))
        jobs = result.scalars().all()
        print([j.id for j in jobs])
if __name__ == '__main__':
    asyncio.run(main())
