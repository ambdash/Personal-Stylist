from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from src.bot.keyboards import main_menu, style_menu
from src.bot.states import BotStates
from src.bot.services.neo4j_service import get_style_recommendations
from src.bot.services.inference_service import generate_recommendation
from src.bot.services.metrics_service import track_request
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
        "3. Ответить на вопросы о моде\n"
        "4. Добавить новые стили и предметы\n\n"
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

@router.message(Command("add_style"))
async def add_style_command(message: Message, state: FSMContext):
    await state.set_state(BotStates.waiting_for_style_name)
    await message.answer("Введите название нового стиля:")

@router.message(BotStates.waiting_for_style_name)
async def process_style_name(message: Message, state: FSMContext):
    style_name = message.text
    await state.update_data(style_name=style_name)
    await state.set_state(BotStates.waiting_for_style_description)
    await message.answer("Введите описание стиля (или отправьте '-' чтобы пропустить):")

@router.message(BotStates.waiting_for_style_description)
async def process_style_description(message: Message, state: FSMContext):
    data = await state.get_data()
    style_name = data['style_name']
    description = None if message.text == '-' else message.text
    
    # Add style to Neo4j
    task = generate_recommendation.delay(style_name, description)
    await message.answer("⏳ Добавляю новый стиль...")
    
    try:
        result = await task.get(timeout=30)
        await message.answer(f"✅ Стиль '{style_name}' успешно добавлен!")
    except Exception as e:
        logger.error(f"Error adding style: {e}")
        await message.answer("❌ Произошла ошибка при добавлении стиля.")
    
    await state.clear()

@router.message(Command("add_item"))
async def add_item_command(message: Message, state: FSMContext):
    await state.set_state(BotStates.waiting_for_item_name)
    await message.answer("Введите название предмета одежды:")

@router.message(BotStates.waiting_for_item_name)
async def process_item_name(message: Message, state: FSMContext):
    item_name = message.text
    await state.update_data(item_name=item_name)
    await state.set_state(BotStates.waiting_for_item_style)
    await message.answer("Выберите стиль для предмета:", reply_markup=style_menu())

@router.callback_query(BotStates.waiting_for_item_style)
async def process_item_style(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    item_name = data['item_name']
    style_name = callback.data.split("_")[1]
    
    # Add item to Neo4j
    task = generate_recommendation.delay(item_name, style_name)
    await callback.message.answer("⏳ Добавляю новый предмет...")
    
    try:
        result = await task.get(timeout=30)
        await callback.message.answer(f"✅ Предмет '{item_name}' успешно добавлен в стиль '{style_name}'!")
    except Exception as e:
        logger.error(f"Error adding item: {e}")
        await callback.message.answer("❌ Произошла ошибка при добавлении предмета.")
    
    await state.clear()
    await callback.answer()

@router.message()
async def handle_message(message: Message):
    """Handle all other messages"""
    await message.answer(
        "Пожалуйста, используйте команды или кнопки меню для взаимодействия со мной.\n"
        "Отправьте /help для списка команд."
    ) 