from aiogram import Router, F
from aiogram.types import Message, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from src.bot.services.neo4j_service import Neo4jService
from src.bot.keyboards import get_node_labels_keyboard, get_nodes_keyboard, get_node_info_keyboard
from src.bot.states import BotStates

router = Router()
neo4j_service = Neo4jService(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password123"
)

class DBExplorerState(StatesGroup):
    browsing_labels = State()
    browsing_nodes = State()
    viewing_node = State()

@router.message(Command("explore_db"))
async def start_db_exploration(message: Message, state: FSMContext):
    """Start database exploration."""
    labels = neo4j_service.get_all_node_labels()
    await message.answer(
        "Choose a node type to explore:",
        reply_markup=get_node_labels_keyboard(labels)
    )
    await state.set_state(DBExplorerState.browsing_labels)

@router.callback_query(F.data.startswith("label_"))
async def show_nodes_of_label(callback: CallbackQuery, state: FSMContext):
    """Show random nodes of selected label."""
    label = callback.data.split("_")[1]
    
    # Store the current label
    await state.update_data(current_label=label, page=0, seen_nodes=[])
    
    # Get random nodes
    nodes = neo4j_service.get_random_nodes_by_label(label)
    
    # Store seen node IDs
    await state.update_data(seen_nodes=[node['id'] for node in nodes])
    
    await callback.message.edit_text(
        f"Showing random nodes of type {label}:",
        reply_markup=get_nodes_keyboard(nodes)
    )
    await state.set_state(DBExplorerState.browsing_nodes)

@router.callback_query(F.data.startswith("page_"))
async def handle_pagination(callback: CallbackQuery, state: FSMContext):
    """Handle pagination of nodes."""
    page = int(callback.data.split("_")[1])
    data = await state.get_data()
    label = data.get('current_label')
    seen_nodes = data.get('seen_nodes', [])
    
    # Get new random nodes, excluding previously seen ones
    nodes = neo4j_service.get_random_nodes_by_label(
        label=label,
        exclude_ids=seen_nodes
    )
    
    # Update seen nodes
    seen_nodes.extend([node['id'] for node in nodes])
    await state.update_data(page=page, seen_nodes=seen_nodes)
    
    await callback.message.edit_text(
        f"Showing more nodes of type {label}:",
        reply_markup=get_nodes_keyboard(nodes, page)
    )

@router.callback_query(F.data.startswith("node_"))
async def show_node_info(callback: CallbackQuery, state: FSMContext):
    """Show detailed information about a node."""
    node_id = callback.data.split("_")[1]
    
    # Get node info and relationships
    node = neo4j_service.get_node_info(node_id)
    relationships = neo4j_service.get_node_relationships(node_id)
    
    if node:
        text = f"Node: {node['name']}\nType: {node['label']}\n\n"
        text += "Related nodes by relationship type:"
        
        await callback.message.edit_text(
            text,
            reply_markup=get_node_info_keyboard(node_id, relationships)
        )
        await state.set_state(DBExplorerState.viewing_node)
    else:
        await callback.answer("Node not found!")

@router.callback_query(F.data == "back_to_labels")
async def back_to_labels(callback: CallbackQuery, state: FSMContext):
    """Return to label selection."""
    labels = neo4j_service.get_all_node_labels()
    await callback.message.edit_text(
        "Choose a node type to explore:",
        reply_markup=get_node_labels_keyboard(labels)
    )
    await state.set_state(DBExplorerState.browsing_labels)

@router.callback_query(F.data == "back_to_nodes")
async def back_to_nodes(callback: CallbackQuery, state: FSMContext):
    """Return to node browsing."""
    data = await state.get_data()
    label = data.get('current_label')
    page = data.get('page', 0)
    
    nodes = neo4j_service.get_random_nodes_by_label(label)
    await callback.message.edit_text(
        f"Showing nodes of type {label}:",
        reply_markup=get_nodes_keyboard(nodes, page)
    )
    await state.set_state(DBExplorerState.browsing_nodes)

@router.message(Command("stop_db_session"))
async def stop_db_session(message: Message, state: FSMContext):
    """Stop database exploration session."""
    await state.clear()
    await message.answer(
        "Database exploration session ended. Use /explore_db to start a new session."
    ) 