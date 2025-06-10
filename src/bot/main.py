import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aiogram import Bot, Dispatcher
from aiogram.types import Message, BotCommand
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from src.bot.states import UnifiedInferenceState
from src.bot.handlers import commands, db_utils_handler, unified_inference_handler
from src.bot.keyboards import get_main_keyboard
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not found in environment variables")

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Register routers
dp.include_router(unified_inference_handler.router)
dp.include_router(commands.router)
dp.include_router(db_utils_handler.router)

async def setup_commands(bot: Bot):
    """Setup bot commands"""
    await bot.set_my_commands([
        BotCommand(command="start", description="Начать работу с ботом"),
        BotCommand(command="ask", description="Задать вопрос о стиле и моде"),
        BotCommand(command="ask_with_params", description="Задать вопрос с параметрами"),
        BotCommand(command="debug_rag", description="Отладка RAG сервиса"),
        BotCommand(command="db_utils", description="Работа с базой данных"),
        BotCommand(command="help", description="Показать справку")
    ])

async def on_startup(bot: Bot):
    """Startup actions"""
    try:
        # Setup commands
        await setup_commands(bot)
        logger.info("Bot commands have been set up")
        logger.info("Bot started successfully")
    except Exception as e:
        logger.error(f"Error in startup: {e}")
        raise

async def on_shutdown(bot: Bot):
    """Shutdown actions"""
    try:
        # Close bot session
        await bot.session.close()
        logger.info("Bot session closed")
    except Exception as e:
        logger.error(f"Error in shutdown: {e}")

async def main():
    """Main function to start the bot"""
    try:
        # Register startup and shutdown handlers
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        
        logger.info("Starting bot...")
        # Run in polling mode
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
            
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main()) 