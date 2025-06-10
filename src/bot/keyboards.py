from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)
from typing import List, Tuple, Optional
from aiogram import Router, F
from aiogram.types import CallbackQuery

router = Router()

def main_menu() -> InlineKeyboardMarkup:
    """Create main menu keyboard"""
    keyboard = [
        [InlineKeyboardButton(text="🔍 Поиск", callback_data="search")],
        [InlineKeyboardButton(text="👗 Стили", callback_data="styles")],
        [InlineKeyboardButton(text="📚 Помощь", callback_data="help")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def style_menu() -> InlineKeyboardMarkup:
    """Create style selection keyboard"""
    styles = [
        "Casual", "Business", "Formal", "Sport",
        "Romantic", "Boho", "Classic", "Street"
    ]
    keyboard = [
        [InlineKeyboardButton(text=style, callback_data=f"style_{style.lower()}")]
        for style in styles
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def event_menu() -> InlineKeyboardMarkup:
    """Event selection menu"""
    keyboard = [
        [
            InlineKeyboardButton(text="🎉 Вечеринка", callback_data="event_party"),
            InlineKeyboardButton(text="💼 Офис", callback_data="event_office")
        ],
        [
            InlineKeyboardButton(text="❤️ Свидание", callback_data="event_date"),
            InlineKeyboardButton(text="🎵 Концерт", callback_data="event_concert")
        ],
        [
            InlineKeyboardButton(text="🚶‍♀️ Прогулка", callback_data="event_walk"),
            InlineKeyboardButton(text="✈️ Путешествие", callback_data="event_travel")
        ],
        [
            InlineKeyboardButton(text="« Назад", callback_data="menu_main")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_node_labels_keyboard(labels: List[Tuple[str, int]]) -> InlineKeyboardMarkup:
    """Create keyboard with node labels and their counts."""
    buttons = []
    for label, count in labels:
        buttons.append(
            InlineKeyboardButton(
                text=f"{label} ({count})",
                callback_data=f"label_{label}"
            )
        )
    return InlineKeyboardMarkup(inline_keyboard=[buttons])

def get_nodes_keyboard(nodes: List[dict], page: int = 0) -> InlineKeyboardMarkup:
    """Create keyboard with node buttons and navigation."""
    buttons = []
    
    # Add node buttons
    for node in nodes:
        buttons.append([
            InlineKeyboardButton(
                text=node['name'],
                callback_data=f"node_{node['id']}"
            )
        ])
    
    # Add navigation row
    nav_buttons = []
    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(
                text="⬅️ Previous",
                callback_data=f"page_{page-1}"
            )
        )
    nav_buttons.append(
        InlineKeyboardButton(
            text="➡️ Next",
            callback_data=f"page_{page+1}"
        )
    )
    buttons.append(nav_buttons)
    
    # Add back button
    buttons.append([
        InlineKeyboardButton(
            text="🔙 Back to Labels",
            callback_data="back_to_labels"
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_node_info_keyboard(node_id: str, relationships: List[dict]) -> InlineKeyboardMarkup:
    """Create keyboard for node information and its relationships."""
    buttons = []
    
    # Group relationships by type
    rel_by_type = {}
    for rel in relationships:
        rel_type = rel['type']
        if rel_type not in rel_by_type:
            rel_by_type[rel_type] = []
        rel_by_type[rel_type].append(rel['node'])
    
    # Add relationship type buttons
    for rel_type, nodes in rel_by_type.items():
        buttons.append([
            InlineKeyboardButton(
                text=f"{rel_type} ({len(nodes)})",
                callback_data=f"rel_{node_id}_{rel_type}"
            )
        ])
    
    # Add back button
    buttons.append([
        InlineKeyboardButton(
            text="🔙 Back",
            callback_data="back_to_nodes"
        )
    ])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_main_keyboard() -> ReplyKeyboardMarkup:
    """Get main menu keyboard"""
    keyboard = [
        [KeyboardButton(text="❓ Задать вопрос")],
        [KeyboardButton(text="🗄 База данных")],
        [KeyboardButton(text="ℹ️ Помощь")]
    ]
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)

def get_inference_type_keyboard() -> InlineKeyboardMarkup:
    """Get keyboard for inference type selection"""
    keyboard = [
        [
            InlineKeyboardButton(text="🤖 Обычный", callback_data="regular_inference"),
            InlineKeyboardButton(text="🧠 Умный", callback_data="rag_inference")
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_db_utils_keyboard() -> InlineKeyboardMarkup:
    """Get keyboard for database operations"""
    keyboard = [
        [InlineKeyboardButton(text="➕ Добавить узел", callback_data="add_node")],
        [InlineKeyboardButton(text="🔍 Поиск по словам", callback_data="search_word")],
        [InlineKeyboardButton(text="🔗 Добавить связь", callback_data="add_relation")],
        [InlineKeyboardButton(text="✏️ Обновить узел", callback_data="update_node")],
        [InlineKeyboardButton(text="🗑 Удалить узел", callback_data="delete_node")],
        [InlineKeyboardButton(text="🔧 Проверить подключение", callback_data="health_check")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_node_types_keyboard(include_other: bool = False) -> InlineKeyboardMarkup:
    """Get keyboard with node types"""
    keyboard = [
        [InlineKeyboardButton(text="Концепт", callback_data="Концепт")],
        [InlineKeyboardButton(text="Эстетика", callback_data="Эстетика")],
        [InlineKeyboardButton(text="Сезон", callback_data="Сезон")],
        [InlineKeyboardButton(text="Случай", callback_data="Случай")],
        [InlineKeyboardButton(text="Тренд", callback_data="Тренд")],
        [InlineKeyboardButton(text="Погода", callback_data="Погода")]
    ]
    
    if include_other:
        keyboard.append([InlineKeyboardButton(text="Другое", callback_data="other")])
    
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_relation_types_keyboard() -> InlineKeyboardMarkup:
    """Get keyboard with relation types"""
    keyboard = [
        [InlineKeyboardButton(text="ОТНОСИТСЯ_К", callback_data="ОТНОСИТСЯ_К")],
        [InlineKeyboardButton(text="ПОДХОДИТ_ДЛЯ", callback_data="ПОДХОДИТ_ДЛЯ")],
        [InlineKeyboardButton(text="В_СЕЗОНЕ", callback_data="В_СЕЗОНЕ")],
        [InlineKeyboardButton(text="🔙 Назад", callback_data="back")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_back_keyboard() -> InlineKeyboardMarkup:
    """Get back button keyboard"""
    keyboard = [[InlineKeyboardButton(text="« Назад", callback_data="back")]]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_parameter_keyboard() -> InlineKeyboardMarkup:
    """Get keyboard for parameter configuration"""
    keyboard = [
        [
            InlineKeyboardButton(text="🎯 Использовать стандартные", callback_data="use_default_params"),
            InlineKeyboardButton(text="⚙️ Настроить", callback_data="configure_params")
        ],
        [InlineKeyboardButton(text="❓ Объяснить параметры", callback_data="explain_params")]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

@router.callback_query(F.data == "back_to_menu")
async def back_to_menu(callback: CallbackQuery):
    """Handle back to menu button"""
    await callback.message.edit_text(
        "🗄 Выберите операцию с базой данных:",
        reply_markup=get_db_utils_keyboard()
    )
    await callback.answer()

@router.callback_query(F.data == "back_to_types")
async def back_to_types(callback: CallbackQuery):
    """Handle back to types button"""
    await callback.message.edit_text(
        "📂 Выберите тип узлов для просмотра:",
        reply_markup=get_node_types_keyboard()
    )
    await callback.answer()

def create_inline_keyboard(buttons: List[List[InlineKeyboardButton]]) -> InlineKeyboardMarkup:
    """Helper function to create inline keyboard with proper initialization"""
    return InlineKeyboardMarkup(inline_keyboard=buttons) 