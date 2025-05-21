from aiogram.fsm.state import State, StatesGroup

class BotStates(StatesGroup):
    waiting_for_style = State()
    waiting_for_event = State()
    waiting_for_prompt = State()
    waiting_for_confirmation = State() 
    waiting_for_style_name = State()
    waiting_for_style_description = State()
    waiting_for_item_name = State()
    waiting_for_item_style = State() 