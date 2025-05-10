from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)

def main_menu() -> ReplyKeyboardMarkup:
    """Main menu keyboard"""
    keyboard = [
        [
            KeyboardButton(text="🎨 Подобрать стиль"),
            KeyboardButton(text="💡 Рекомендации")
        ],
        [
            KeyboardButton(text="📊 Статистика"),
            KeyboardButton(text="❓ Помощь")
        ]
    ]
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        input_field_placeholder="Выберите действие"
    )

def style_menu() -> InlineKeyboardMarkup:
    """Style selection menu"""
    keyboard = [
        [
            InlineKeyboardButton(text="Y2K", callback_data="style_y2k"),
            InlineKeyboardButton(text="Гранж", callback_data="style_grunge")
        ],
        [
            InlineKeyboardButton(text="Old Money", callback_data="style_oldmoney"),
            InlineKeyboardButton(text="Clean Girl", callback_data="style_cleangirl")
        ],
        [
            InlineKeyboardButton(text="Dark Academia", callback_data="style_darkacademia"),
            InlineKeyboardButton(text="Cottagecore", callback_data="style_cottagecore")
        ],
        [
            InlineKeyboardButton(text="« Назад", callback_data="menu_main")
        ]
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