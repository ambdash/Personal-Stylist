from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from ..keyboards import main_menu, style_menu
from ..states import BotStates
from ..services.neo4j_service import get_style_recommendations
from ..services.inference_service import generate_recommendation
from ..services.metrics_service import track_request
from celery.result import AsyncResult
import logging

router = Router()
logger = logging.getLogger(__name__)

@router.message(Command("start"))
async def start_command(message: Message):
    await message.answer(
        "👋 Привет! Я твой персональный AI-стилист. Я помогу тебе:\n"
        "1. Подобрать образ\n"
        "2. Дать рекомендации по стилю\n"
        "3. Ответить на вопросы о моде\n\n"
        "Выберите действие:",
        reply_markup=main_menu()
    )

@router.message(Command("help"))
async def help_command(message: Message):
    await message.answer(
        "🤖 Вот что я умею:\n\n"
        "📱 Основные команды:\n"
        "/start - Начать работу\n"
        "/style - Подобрать стиль\n"
        "/recommend - Получить рекомендации\n"
        "/history - История запросов\n\n"
        "💡 Также вы можете:\n"
        "- Написать свой запрос\n"
        "- Выбрать действие из меню\n"
        "- Получить статистику использования"
    )

@router.message(Command("style"))
async def style_command(message: Message, state: FSMContext):
    await state.set_state(BotStates.waiting_for_style)
    await message.answer(
        "Выберите стиль, который вас интересует:",
        reply_markup=style_menu()
    )

@router.callback_query(F.data.startswith("style_"))
async def process_style_selection(callback: CallbackQuery, state: FSMContext):
    style = callback.data.split("_")[1]
    await state.update_data(selected_style=style)
    
    # Get recommendations from Neo4j
    recommendations = await get_style_recommendations(style)
    
    # Track request in Prometheus
    track_request("style_recommendation", style)
    
    # Format recommendations
    response = f"🎨 Рекомендации для стиля {style}:\n\n"
    for rec in recommendations:
        response += f"• {rec}\n"
    
    await callback.message.answer(response)
    await callback.answer()

@router.message(Command("recommend"))
async def recommend_command(message: Message, state: FSMContext):
    await state.set_state(BotStates.waiting_for_prompt)
    await message.answer(
        "Опишите, для какого случая вам нужны рекомендации.\n"
        "Например: 'Нужен образ для свидания в ресторане'"
    )

@router.message(BotStates.waiting_for_prompt)
async def process_prompt(message: Message, state: FSMContext):
    # Add task to Celery queue
    task = generate_recommendation.delay(message.text)
    
    # Store task ID in state
    await state.update_data(task_id=task.id)
    await message.answer("⏳ Генерирую рекомендации...")
    
    # Wait for result
    result = AsyncResult(task.id)
    try:
        recommendation = await result.get(timeout=30)
        await message.answer(recommendation)
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        await message.answer("Извините, произошла ошибка. Попробуйте позже.")
    
    await state.clear()

@router.message()
async def handle_message(message: Message):
    """Handle all other messages"""
    await message.answer(
        "Пожалуйста, используйте команды или кнопки меню для взаимодействия со мной.\n"
        "Отправьте /help для списка команд."
    ) 