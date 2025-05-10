import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def async_retry(retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error(f"Failed after {retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(delay * (attempt + 1))
            return await func(*args, **kwargs)
        return wrapper
    return decorator 