from aiogram.fsm.state import State, StatesGroup

class BotStates(StatesGroup):
    """Basic bot states"""
    waiting_for_style = State()
    waiting_for_prompt = State()
    waiting_for_style_name = State()
    waiting_for_style_description = State()
    waiting_for_item_name = State()
    waiting_for_item_style = State()

class UnifiedInferenceState(StatesGroup):
    """States for unified inference process"""
    choosing_type = State()
    waiting_for_prompt = State()
    waiting_for_result = State()

class DbUtilsState(StatesGroup):
    """Database operation states"""
    waiting_for_action = State()
    waiting_for_node_name = State()
    waiting_for_node_label = State()
    waiting_for_node_properties = State()
    waiting_for_search_word = State()
    waiting_for_start_node = State()
    waiting_for_end_node = State()
    waiting_for_relation_type = State()
    waiting_for_delete_confirm = State()
    waiting_for_update_node = State()
    waiting_for_update_field = State()
    waiting_for_update_value = State()
    waiting_for_node_type = State()
    waiting_for_new_node_type = State()

class RagState(StatesGroup):
    waiting_for_prompt = State() 