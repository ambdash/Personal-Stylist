from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from src.bot.states import DbUtilsState
from src.bot.keyboards import (
    get_db_utils_keyboard,
    get_node_types_keyboard,
    get_relation_types_keyboard,
    get_back_keyboard,
    create_inline_keyboard
)
from aiogram.types import InlineKeyboardButton
from src.celery_db_app import app as celery_app
import logging
import asyncio
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
router = Router()

async def wait_for_task_completion(task_id: str, timeout: int = 120) -> Dict[str, Any]:
    """Wait for Celery task completion with timeout"""
    task = celery_app.AsyncResult(task_id)
    
    # Wait for completion with timeout
    for _ in range(timeout):
        if task.state == 'SUCCESS':
            return {
                "status": "completed",
                "result": task.result
            }
        elif task.state == 'FAILURE':
            return {
                "status": "failed",
                "error": str(task.info)
            }
        elif task.state in ['PENDING', 'STARTED', 'RETRY']:
            await asyncio.sleep(1)
        else:
            return {
                "status": "unknown",
                "state": task.state
            }
    
    return {
        "status": "timeout",
        "error": "Task timed out"
    }

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
        "• Обновить узел - изменение свойств узла\n"
        "• Удалить узел - удаление узла\n"
        "• Проверить подключение - тест соединения с базой данных",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_action)

@router.callback_query(DbUtilsState.waiting_for_action)
async def handle_db_action(callback: CallbackQuery, state: FSMContext):
    """Handle database action selection"""
    action = callback.data
    await callback.answer()
    
    if action == "add_node":
        await state.set_state(DbUtilsState.waiting_for_node_name)
        await callback.message.edit_text(
            "📝 Введите название нового узла:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "search_style":
        await state.set_state(DbUtilsState.waiting_for_search_word)
        await callback.message.edit_text(
            "🔍 Введите стиль для поиска:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "search_word":
        await state.set_state(DbUtilsState.waiting_for_search_word)
        await callback.message.edit_text(
            "🔍 Введите слово для поиска:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "add_relation":
        await state.set_state(DbUtilsState.waiting_for_start_node)
        await callback.message.edit_text(
            "🔗 Введите название первого узла:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "update_node":
        await state.set_state(DbUtilsState.waiting_for_node_name)
        await callback.message.edit_text(
            "✏️ Введите название узла для обновления:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "delete_node":
        await state.set_state(DbUtilsState.waiting_for_node_name)
        await callback.message.edit_text(
            "🗑 Введите название узла для удаления:",
            reply_markup=get_back_keyboard()
        )
    
    elif action == "health_check":
        # Test database connection
        processing_msg = await callback.message.edit_text("🔄 Проверяю подключение к базе данных...")
        
        try:
            # Submit health check task
            task = celery_app.send_task(
                'db_worker.health_check',
                queue='database'
            )
            
            # Wait for result
            result = await wait_for_task_completion(task.id, timeout=30)
            
            if result["status"] == "completed":
                health_result = result["result"]
                if health_result.get("success"):
                    await processing_msg.edit_text(
                        f"✅ Подключение к базе данных работает!\n\n"
                        f"📊 Информация:\n"
                        f"• Соединение: {health_result.get('connection', 'N/A')}\n"
                        f"• Статус: {health_result.get('message', 'OK')}",
                        reply_markup=get_db_utils_keyboard()
                    )
                else:
                    await processing_msg.edit_text(
                        f"❌ Ошибка подключения к базе данных:\n{health_result.get('error', 'Unknown error')}",
                        reply_markup=get_db_utils_keyboard()
                    )
            else:
                await processing_msg.edit_text(
                    f"❌ Не удалось проверить подключение: {result.get('error', 'Unknown error')}",
                    reply_markup=get_db_utils_keyboard()
                )
        except Exception as e:
            await processing_msg.edit_text(
                f"❌ Ошибка при проверке подключения: {str(e)}",
                reply_markup=get_db_utils_keyboard()
            )

@router.message(DbUtilsState.waiting_for_node_name)
async def process_node_name(message: Message, state: FSMContext):
    """Process node name input"""
    await state.update_data(node_name=message.text)
    keyboard = get_node_types_keyboard()
    await message.answer(
        "📋 Выберите тип узла:",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_node_label)

@router.callback_query(DbUtilsState.waiting_for_node_label)
async def process_node_label(callback: CallbackQuery, state: FSMContext):
    """Process node label selection"""
    data = await state.get_data()
    node_name = data["node_name"]
    node_type = callback.data
    
    processing_msg = await callback.message.edit_text("🔄 Создаю узел...")
    
    try:
        # Submit create node task
        task = celery_app.send_task(
            'db_worker.create_node',
            args=[node_type, {"name": node_name, "id": f"{node_type.lower()}_{node_name.lower().replace(' ', '_')}"}],
            queue='database'
        )
        
        # Wait for result
        result = await wait_for_task_completion(task.id)
        
        if result["status"] == "completed":
            task_result = result["result"]
            if task_result.get("success"):
                await processing_msg.edit_text(
                    f"✅ Узел успешно создан!\n"
                    f"Название: {node_name}\n"
                    f"Тип: {node_type}",
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await processing_msg.edit_text(
                    f"❌ Ошибка при создании узла: {task_result.get('error', 'Unknown error')}",
                    reply_markup=get_db_utils_keyboard()
                )
        else:
            await processing_msg.edit_text(
                f"❌ Не удалось создать узел: {result.get('error', 'Unknown error')}",
                reply_markup=get_db_utils_keyboard()
            )
    except Exception as e:
        await processing_msg.edit_text(
            f"❌ Ошибка при создании узла: {str(e)}",
            reply_markup=get_db_utils_keyboard()
        )
    
    await state.clear()

@router.message(DbUtilsState.waiting_for_search_word)
async def process_search_word(message: Message, state: FSMContext):
    """Process search word input"""
    search_query = message.text.strip()
    processing_msg = await message.answer("🔄 Ищу в базе данных...")
    
    try:
        # Use HTTP API instead of Celery to avoid SIGSEGV crashes
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:8000/telegram/db/search",
                json={"word": search_query, "limit": 20},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                nodes = result.get("nodes", [])
                
                if nodes:
                    response_text = f"🔍 Найдено {len(nodes)} результатов:\n\n"
                    for i, node in enumerate(nodes[:10], 1):  # Show first 10 results
                        response_text += f"{i}. {node.get('name', 'Unnamed')}\n"
                        if node.get('total_connections', 0) > 0:
                            response_text += f"   Связей: {node['total_connections']}\n"
                        response_text += "\n"
                    
                    if len(nodes) > 10:
                        response_text += f"... и еще {len(nodes) - 10} результатов"
                else:
                    response_text = "😕 Ничего не найдено. Попробуйте другое слово."
                
                await processing_msg.edit_text(
                    response_text,
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await processing_msg.edit_text(
                    f"❌ Ошибка поиска: HTTP {response.status_code}",
                    reply_markup=get_db_utils_keyboard()
                )
                
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        await processing_msg.edit_text(
            f"❌ Ошибка при поиске: {str(e)}",
            reply_markup=get_db_utils_keyboard()
        )
    
    await state.clear()

@router.message(DbUtilsState.waiting_for_start_node)
async def handle_start_node(message: Message, state: FSMContext):
    """Handle start node for relation"""
    start_node = message.text.strip()
    await state.update_data(start_node=start_node)
    
    await message.answer(
        "🔗 Введите название второго узла:",
        reply_markup=get_back_keyboard()
    )
    await state.set_state(DbUtilsState.waiting_for_end_node)

@router.message(DbUtilsState.waiting_for_end_node)
async def process_end_node(message: Message, state: FSMContext):
    """Process end node for relation"""
    end_node = message.text.strip()
    await state.update_data(end_node=end_node)
    
    # Show relation type options
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="ОТНОСИТСЯ_К", callback_data="ОТНОСИТСЯ_К")],
        [InlineKeyboardButton(text="ПОДХОДИТ_ДЛЯ", callback_data="ПОДХОДИТ_ДЛЯ")],
        [InlineKeyboardButton(text="В_СЕЗОНЕ", callback_data="В_СЕЗОНЕ")],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back")]
    ])
    
    await message.answer(
        "🔗 Выберите тип связи:",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_relation_type)

@router.callback_query(DbUtilsState.waiting_for_relation_type)
async def process_relation_type(callback: CallbackQuery, state: FSMContext):
    """Process relation type selection"""
    relation_type = callback.data
    data = await state.get_data()
    
    processing_msg = await callback.message.edit_text("🔄 Создаю связь...")
    
    try:
        # Submit create relationship task
        task = celery_app.send_task(
            'db_worker.create_relationship',
            args=[
                f"{data['start_node'].lower().replace(' ', '_')}",
                f"{data['end_node'].lower().replace(' ', '_')}",
                relation_type
            ],
            queue='database'
        )
        
        # Wait for result
        result = await wait_for_task_completion(task.id)
        
        if result["status"] == "completed":
            task_result = result["result"]
            if task_result.get("success"):
                await processing_msg.edit_text(
                    f"✅ Связь успешно создана!\n"
                    f"{data['start_node']} → {relation_type} → {data['end_node']}",
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await processing_msg.edit_text(
                    f"❌ Ошибка при создании связи: {task_result.get('error', 'Unknown error')}",
                    reply_markup=get_db_utils_keyboard()
                )
        else:
            await processing_msg.edit_text(
                f"❌ Не удалось создать связь: {result.get('error', 'Unknown error')}",
                reply_markup=get_db_utils_keyboard()
            )
    except Exception as e:
        await processing_msg.edit_text(
            f"❌ Ошибка при создании связи: {str(e)}",
            reply_markup=get_db_utils_keyboard()
        )
    
    await state.clear()

@router.callback_query(F.data == "back_to_menu")
async def back_to_menu(callback: CallbackQuery, state: FSMContext):
    """Return to main menu"""
    await callback.answer()
    await state.clear()
    await callback.message.edit_text(
        "🗄 Выберите операцию с базой данных:",
        reply_markup=get_db_utils_keyboard()
    )
    await state.set_state(DbUtilsState.waiting_for_action)

@router.callback_query(F.data == "back")
async def process_back(callback: CallbackQuery, state: FSMContext):
    """Handle back button"""
    await callback.answer()
    await state.clear()
    await callback.message.edit_text(
        "🗄 Выберите операцию с базой данных:",
        reply_markup=get_db_utils_keyboard()
    )
    await state.set_state(DbUtilsState.waiting_for_action)