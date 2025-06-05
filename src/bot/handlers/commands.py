from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, BotCommand
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from src.bot.keyboards import get_main_keyboard, get_inference_type_keyboard, get_db_utils_keyboard, get_node_types_keyboard
from src.bot.states import UnifiedInferenceState, DbUtilsState
from src.bot.services.neo4j_service import search_nodes_by_word, get_style_recommendations, get_node_by_type, add_node
from src.bot.services.metrics_service import track_request
import logging

router = Router()
logger = logging.getLogger(__name__)

async def setup_bot_commands(bot):
    """Setup bot commands in the menu"""
    commands = [
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="inference_async", description="Запрос к LLM модели"),
        BotCommand(command="rag_inference_async", description="Запрос к LLM модели с использованием базы знаний"),
        BotCommand(command="db_utils", description="Работа с базой данных"),
        BotCommand(command="help", description="Получить справку по командам"),
    ]
    await bot.set_my_commands(commands)

@router.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command"""
    await message.answer(
        "👋 Привет! Я ваш персональный помощник по стилю.\n\n"
        "🤖 Вот что я умею:\n\n"
        "📱 Основные команды:\n"
        "/ask - Задать вопрос о стиле и моде\n"
        "/db_utils - Работа с базой данных\n"
        "/help - Получить справку\n\n"
        "💡 Выберите команду из меню или напишите /help для подробной информации.",
        reply_markup=get_main_keyboard()
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command"""
    help_text = (
        "📚 Справка по командам:\n\n"
        "1️⃣ /ask\n"
        "   • Задать вопрос о стиле и моде\n"
        "   • Получить персональные рекомендации\n\n"
        "2️⃣ /db_utils\n"
        "   • Инструменты для работы с базой данных\n"
        "   • Поиск узлов по словам\n"
        "   • Просмотр связей между элементами\n"
        "   • Добавление новых узлов\n\n"
        "💡 Совет: Используйте /ask для получения рекомендаций"
    )
    await message.answer(help_text)

@router.message(Command("ask"))
async def cmd_ask(message: Message, state: FSMContext):
    """Handle /ask command"""
    keyboard = get_inference_type_keyboard()
    await message.answer(
        "🤖 Выберите режим запроса:\n\n"
        "• Обычный - использует только модель для генерации ответа\n"
        "• Умный - дополнительно использует базу знаний для более точного ответа",
        reply_markup=keyboard
    )
    await state.set_state(UnifiedInferenceState.choosing_type)

@router.message(Command("db_utils"))
async def cmd_db_utils(message: Message, state: FSMContext):
    """Handle /db_utils command"""
    keyboard = get_db_utils_keyboard()
    await message.answer(
        "🗄 Выберите операцию с базой данных:\n\n"
        "• Добавить узел - создание нового узла\n"
        "• Поиск по стилю - поиск узлов определенного стиля\n"
        "• Поиск по словам - текстовый поиск по узлам\n"
        "• Добавить связь - создание связи между узлами\n"
        "• Обновить узел - изменение свойств узла",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_action)

@router.callback_query(DbUtilsState.waiting_for_action)
async def process_db_action(callback: CallbackQuery, state: FSMContext):
    """Process database action selection"""
    action = callback.data
    
    if action == "search_word":
        await callback.message.answer(
            "🔍 Введите слово для поиска в базе данных:"
        )
        await state.set_state(DbUtilsState.waiting_for_search_word)
    
    elif action == "view_by_type":
        await callback.message.answer(
            "📂 Выберите тип узлов для просмотра:",
            reply_markup=get_node_types_keyboard()
        )
        await state.set_state(DbUtilsState.waiting_for_node_type)
    
    elif action == "add_node":
        await callback.message.answer(
            "➕ Введите название нового узла:"
        )
        await state.set_state(DbUtilsState.waiting_for_node_name)
    
    await callback.answer()

@router.message(DbUtilsState.waiting_for_search_word)
async def process_search_word(message: Message, state: FSMContext):
    """Process word search in database"""
    word = message.text.strip()
    result = await search_nodes_by_word(word)
    
    if not result["found"]:
        await message.answer(
            "😕 Ничего не найдено. Попробуйте другое слово."
        )
        return
    
    response = "🔍 Результаты поиска:\n\n"
    for node in result["nodes"]:
        response += f"📌 {node['name']}\n"
        if node.get('connections'):
            response += "   Связи:\n"
            for conn in node['connections']:
                response += f"   • {conn}\n"
    
    await message.answer(response)
    await state.clear()

@router.message(DbUtilsState.waiting_for_node_type)
async def process_node_type(message: Message, state: FSMContext):
    """Process node type input"""
    node_type = message.text.strip()
    nodes = await get_node_by_type(node_type)
    
    if not nodes:
        await message.answer(
            f"😕 Не найдено узлов типа '{node_type}'."
        )
        return
    
    response = f"📂 Узлы типа '{node_type}':\n\n"
    for node in nodes:
        response += f"• {node['name']}\n"
    
    await message.answer(response)
    await state.clear()

@router.message(DbUtilsState.waiting_for_node_name)
async def process_node_name(message: Message, state: FSMContext):
    """Process new node name"""
    name = message.text.strip()
    await state.update_data(node_name=name)
    
    await message.answer(
        "📝 Выберите тип узла:",
        reply_markup=get_node_types_keyboard(include_other=True)
    )
    await state.set_state(DbUtilsState.waiting_for_new_node_type)

@router.callback_query(DbUtilsState.waiting_for_new_node_type)
async def process_new_node_type_callback(callback: CallbackQuery, state: FSMContext):
    """Process new node type from callback"""
    node_type = callback.data
    data = await state.get_data()
    node_name = data.get("node_name")
    
    if node_type not in ["Концепт", "Эстетика", "Сезон", "Случай", "Тренд", "Погода"]:
        await callback.message.answer(
            "❌ Неверный тип узла. Пожалуйста, выберите тип из списка."
        )
        return
    
    result = await add_node(node_name, node_type)
    
    if not result["success"]:
        await callback.message.answer(
            f"❌ {result.get('message', 'Произошла ошибка при добавлении узла.')}"
        )
    else:
        await callback.message.answer(
            f"✅ Узел '{node_name}' типа '{node_type}' успешно добавлен!"
        )
    
    await callback.answer()
    await state.clear()

# Handler for unknown commands
@router.message(lambda message: message.text and message.text.startswith('/'))
async def handle_unknown_command(message: Message):
    """Handle any unrecognized command"""
    await message.answer(
        "❓ Неизвестная команда.\n"
        "Используйте /help, чтобы узнать список доступных команд."
    )

# Default message handler
@router.message()
async def handle_default_message(message: Message):
    """Handle any non-command message"""
    await message.answer(
        "👋 Пожалуйста, используйте команды из меню или отправьте /help для списка команд.",
        reply_markup=get_main_keyboard()
    ) 