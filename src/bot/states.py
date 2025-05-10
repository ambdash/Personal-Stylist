from aiogram.fsm.state import State, StatesGroup

class BotStates(StatesGroup):
    waiting_for_style = State()
    waiting_for_event = State()
    waiting_for_prompt = State()
    waiting_for_confirmation = State() 