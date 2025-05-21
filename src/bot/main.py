import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import RedisStorage
from dotenv import load_dotenv
from src.bot.handlers import commands
from prometheus_client import start_http_server
import os

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis_password")

async def main():
    try:
        # Start Prometheus metrics server on a different port to avoid conflicts with API
        start_http_server(8001)
        
        # Initialize Redis storage with authentication
        storage = RedisStorage.from_url(
            f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/0",
            connection_kwargs={"retry_on_timeout": True}
        )
        
        # Initialize bot and dispatcher
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        dp = Dispatcher(storage=storage)
        
        # Register handlers
        dp.include_router(commands.router)
        
        # Start polling
        logger.info("Starting bot...")
        await dp.start_polling(bot, allowed_updates=["message", "callback_query"])
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 