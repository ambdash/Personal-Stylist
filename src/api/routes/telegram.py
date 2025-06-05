from fastapi import APIRouter, HTTPException, Request, Depends
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from src.ml.inference.engine import InferenceEngine
from src.api.services.kafka_producer import KafkaMessageProducer
import os
import json
import logging
from typing import Optional

router = APIRouter(prefix="/v1/telegram", tags=["telegram"])
logger = logging.getLogger(__name__)

WEBHOOK_PATH = "/v1/telegram/webhook"
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Should be set in production
WEBAPP_HOST = os.getenv("WEBAPP_HOST", "0.0.0.0")
WEBAPP_PORT = int(os.getenv("WEBAPP_PORT", 8000))

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
inference_engine = InferenceEngine()
kafka_producer = KafkaMessageProducer()

@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer(
        "👋 Привет! Я твой персональный стилист. Я помогу тебе:\n"
        "1. Подобрать образ\n"
        "2. Дать рекомендации по стилю\n"
        "3. Ответить на вопросы о моде\n\n"
        "Просто напиши свой вопрос или используй команду /help для подсказок."
    )

@dp.message(Command("help"))
async def help_command(message: types.Message):
    await message.answer(
        "🤖 Вот что я умею:\n\n"
        "/style - Подобрать образ\n"
        "/recommend - Получить рекомендации\n"
        "/trends - Узнать о трендах\n"
        "\nИли просто напиши свой вопрос!"
    )


@router.post("/set-webhook")
async def set_webhook(url: Optional[str] = None):
    """Set up webhook for the bot"""
    try:
        webhook_url = url or WEBHOOK_URL
        if not webhook_url:
            raise HTTPException(
                status_code=400,
                detail="Webhook URL is not provided and not set in environment"
            )
        
        webhook_info = await bot.get_webhook_info()
        if webhook_info.url == webhook_url:
            return {"status": "ok", "message": "Webhook is already set to this URL"}
            
        await bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]
        )
        return {"status": "ok", "message": "Webhook set successfully"}
    except Exception as e:
        logger.error(f"Error setting webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/webhook")
async def delete_webhook():
    """Remove webhook"""
    try:
        await bot.delete_webhook()
        return {"status": "ok", "message": "Webhook deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/webhook-info")
async def get_webhook_info():
    """Get current webhook information"""
    try:
        webhook_info = await bot.get_webhook_info()
        return {
            "url": webhook_info.url,
            "has_custom_certificate": webhook_info.has_custom_certificate,
            "pending_update_count": webhook_info.pending_update_count,
            "last_error_date": webhook_info.last_error_date,
            "last_error_message": webhook_info.last_error_message,
            "max_connections": webhook_info.max_connections,
            "ip_address": webhook_info.ip_address
        }
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await message.answer("Извините, произошла ошибка. Попробуйте позже.")

@router.post("/webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram webhook requests"""
    try:
        data = await request.json()
        update = types.Update(**data)
        await dp.feed_update(bot=bot, update=update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 