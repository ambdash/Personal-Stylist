from fastapi import APIRouter, HTTPException, Request
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from src.ml.inference.engine import InferenceEngine
from src.api.services.kafka_producer import KafkaMessageProducer
import os
import json
import logging

router = APIRouter(prefix="/v1/telegram", tags=["telegram"])
logger = logging.getLogger(__name__)

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

@dp.message()
async def handle_message(message: types.Message):
    ## STUB ADD KAFKA (OR MAYBE REDIS)
    try:
        # Log user message to Kafka
        await kafka_producer.send_message(
            "user_messages",
            {
                "user_id": message.from_user.id,
                "message": message.text,
                "timestamp": message.date.isoformat()
            }
        )

        response, _ = inference_engine.generate(message.text)

        await kafka_producer.send_message(
            "bot_responses",
            {
                "user_id": message.from_user.id,
                "message": response,
                "timestamp": message.date.isoformat()
            }
        )

        await message.answer(response)

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