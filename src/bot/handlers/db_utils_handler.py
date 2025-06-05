from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from src.bot.states import DbUtilsState
from src.api.db.neo4j.service import Neo4jService
from src.bot.keyboards import (
    get_db_utils_keyboard,
    get_node_types_keyboard,
    get_back_keyboard,
    create_inline_keyboard
)
from aiogram.types import InlineKeyboardButton
from src.api.tasks.db_tasks import (
    create_node,
    update_node,
    delete_node,
    create_relation,
    search_nodes,
    search_by_text
)
import logging
import aiohttp
from typing import Dict, Any, Optional
import asyncio
import json

logger = logging.getLogger(__name__)
router = Router()

API_BASE_URL = "http://api:8002"  # Updated to use service name and correct port

async def check_task_status(task_id: str) -> dict:
    """Check the status of a task"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/tasks/{task_id}") as response:
            return await response.json()

def get_neo4j_service():
    """Get Neo4j service instance"""
    return Neo4jService()

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
        "• Удалить узел - удаление узла",
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
    node_data = {
        "name": data["node_name"],
        "label": callback.data,
        "aliases": [],
        "properties": {}
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{API_BASE_URL}/db/nodes/create",
            json=node_data
        ) as response:
            result = await response.json()
            task_id = result.get("task_id")
            
            # Wait for task completion
            task_result = await check_task_status(task_id)
            if task_result.get("status") == "completed":
                await callback.message.edit_text(
                    f"✅ Узел успешно создан!\n"
                    f"Название: {node_data['name']}\n"
                    f"Тип: {node_data['label']}",
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await callback.message.edit_text(
                    "❌ Ошибка при создании узла",
                    reply_markup=get_db_utils_keyboard()
                )
    
    await state.clear()

@router.message(DbUtilsState.waiting_for_search_word)
async def process_search_word(message: Message, state: FSMContext):
    """Process search word input"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/db/search",
            params={"query": message.text}
        ) as response:
            result = await response.json()
            task_id = result.get("task_id")
            
            # Wait for task completion
            task_result = await check_task_status(task_id)
            if task_result.get("status") == "completed":
                nodes = task_result.get("result", {}).get("nodes", [])
                if nodes:
                    response_text = "🔍 Результаты поиска:\n\n"
                    for node in nodes:
                        response_text += f"• {node['name']} ({', '.join(node['labels'])})\n"
                        if node.get('outgoing_relations'):
                            response_text += "  Связи:\n"
                            for rel in node['outgoing_relations']:
                                response_text += f"  → {rel['type']} → {rel['target']}\n"
                else:
                    response_text = "❌ Ничего не найдено"
                
                await message.answer(
                    response_text,
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await message.answer(
                    "❌ Ошибка при поиске",
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
    """Process end node input"""
    end_node = message.text.strip()
    await state.update_data(end_node=end_node)
    
    # Predefined relation types
    relation_types = [
        ["ПОДХОДИТ_ДЛЯ", "ОТНОСИТСЯ_К"],
        ["В_СЕЗОНЕ", "СОЧЕТАЕТСЯ_С"]
    ]
    keyboard = create_inline_keyboard([
        [InlineKeyboardButton(text=rel, callback_data=f"rel_{rel}")]
        for row in relation_types
        for rel in row
    ])
    
    await message.answer(
        "🔗 Выберите тип связи:",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_relation_type)

@router.callback_query(DbUtilsState.waiting_for_relation_type)
async def process_relation_type(callback: CallbackQuery, state: FSMContext):
    """Process relation type selection"""
    rel_type = callback.data.replace("rel_", "")
    data = await state.get_data()
    
    task = create_relation.delay(
        start_node=data['start_node'],
        end_node=data['end_node'],
        rel_type=rel_type
    )
    
    await callback.message.edit_text(
        f"✅ Создание связи между '{data['start_node']}' и '{data['end_node']}' "
        f"типа '{rel_type}' запущено.\n"
        f"ID задачи: {task.id}",
        reply_markup=get_back_keyboard()
    )
    await state.clear()

@router.callback_query(F.data == "back_to_menu")
async def back_to_menu(callback: CallbackQuery, state: FSMContext):
    """Handle back to menu button"""
    await state.clear()
    await callback.message.edit_text(
        "🗄 Выберите операцию с базой данных:",
        reply_markup=get_db_utils_keyboard()
    )
    await callback.answer()

@router.callback_query(F.data == "delete_node")
async def process_delete_request(callback: CallbackQuery, state: FSMContext):
    """Handle delete node request"""
    await callback.message.edit_text(
        "Введите ID узла для удаления:",
        reply_markup=get_back_keyboard()
    )
    await state.set_state(DbUtilsState.waiting_for_delete_confirm)

@router.message(DbUtilsState.waiting_for_delete_confirm)
async def process_delete_confirm(message: Message, state: FSMContext):
    """Process delete confirmation"""
    node_id = message.text
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"{API_BASE_URL}/db/nodes/{node_id}"
        ) as response:
            result = await response.json()
            task_id = result.get("task_id")
            
            # Wait for task completion
            task_result = await check_task_status(task_id)
            if task_result.get("status") == "completed":
                await message.answer(
                    f"✅ Узел {node_id} успешно удален",
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await message.answer(
                    "❌ Ошибка при удалении узла",
                    reply_markup=get_db_utils_keyboard()
                )
    
    await state.clear()

@router.callback_query(F.data == "back")
async def process_back(callback: CallbackQuery, state: FSMContext):
    """Handle back button"""
    await callback.message.edit_text(
        "Выберите операцию с базой данных:",
        reply_markup=get_db_utils_keyboard()
    )
    await state.clear()

@router.callback_query(F.data == "update_node")
async def process_update_request(callback: CallbackQuery, state: FSMContext):
    """Handle update node request"""
    await callback.message.edit_text(
        "Введите ID узла для обновления:",
        reply_markup=get_back_keyboard()
    )
    await state.set_state(DbUtilsState.waiting_for_update_node)

@router.message(DbUtilsState.waiting_for_update_node)
async def process_update_node(message: Message, state: FSMContext):
    """Process node ID for update"""
    await state.update_data(update_node_id=message.text)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Название", callback_data="update_name")],
        [InlineKeyboardButton(text="Алиасы", callback_data="update_aliases")],
        [InlineKeyboardButton(text="Описание", callback_data="update_description")],
        [InlineKeyboardButton(text="« Назад", callback_data="back")]
    ])
    await message.answer(
        "Что хотите обновить?",
        reply_markup=keyboard
    )
    await state.set_state(DbUtilsState.waiting_for_update_field)

@router.callback_query(DbUtilsState.waiting_for_update_field)
async def process_update_field(callback: CallbackQuery, state: FSMContext):
    """Process update field selection"""
    field = callback.data.replace("update_", "")
    await state.update_data(update_field=field)
    
    if field == "aliases":
        await callback.message.edit_text(
            "Введите алиасы через запятую (например: 'найк белый, nike white'):",
            reply_markup=get_back_keyboard()
        )
    elif field == "description":
        await callback.message.edit_text(
            "Введите новое описание:",
            reply_markup=get_back_keyboard()
        )
    else:
        await callback.message.edit_text(
            "Введите новое значение:",
            reply_markup=get_back_keyboard()
        )
    
    await state.set_state(DbUtilsState.waiting_for_update_value)

@router.message(DbUtilsState.waiting_for_update_value)
async def process_update_value(message: Message, state: FSMContext):
    """Process update value input"""
    data = await state.get_data()
    node_id = data["update_node_id"]
    field = data["update_field"]
    value = message.text
    
    update_data = {"properties": {}}
    if field == "name":
        update_data["name"] = value
    elif field == "aliases":
        update_data["aliases"] = [alias.strip() for alias in value.split(",")]
    elif field == "description":
        update_data["properties"]["description"] = value
    
    async with aiohttp.ClientSession() as session:
        async with session.put(
            f"{API_BASE_URL}/db/nodes/{node_id}",
            json=update_data
        ) as response:
            result = await response.json()
            task_id = result.get("task_id")
            
            # Wait for task completion
            task_result = await check_task_status(task_id)
            if task_result.get("status") == "completed":
                await message.answer(
                    f"✅ Узел успешно обновлен!\n"
                    f"Поле: {field}\n"
                    f"Новое значение: {value}",
                    reply_markup=get_db_utils_keyboard()
                )
            else:
                await message.answer(
                    "❌ Ошибка при обновлении узла",
                    reply_markup=get_db_utils_keyboard()
                )
    
    await state.clear()

@router.callback_query(F.data == "check_tasks")
async def process_check_tasks(callback: CallbackQuery, state: FSMContext):
    """Handle task status check"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/health") as response:
            health = await response.json()
            status_text = (
                "📊 Статус сервисов:\n\n"
                f"Neo4j: {'✅' if health.get('services', {}).get('neo4j') == 'connected' else '❌'}\n"
                f"Redis: {'✅' if health.get('services', {}).get('redis') == 'connected' else '❌'}\n"
                f"Celery: {'✅' if health.get('services', {}).get('celery') == 'connected' else '❌'}"
            )
            await callback.message.edit_text(
                status_text,
                reply_markup=get_db_utils_keyboard()
            )