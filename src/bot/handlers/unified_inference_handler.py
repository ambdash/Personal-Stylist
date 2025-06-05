from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext
from src.bot.states import UnifiedInferenceState
from src.bot.keyboards import get_inference_type_keyboard
import aiohttp
import logging
import json
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)
router = Router()

# API configuration
API_BASE_URL = "http://localhost:8000/v1/inference"  # Update with your actual API URL

@router.message(Command("ask"))
async def cmd_unified_inference(message: Message, state: FSMContext):
    """Handle /ask command"""
    keyboard = get_inference_type_keyboard()
    await state.set_state(UnifiedInferenceState.choosing_type)
    await message.answer(
        "🤖 Выберите режим запроса:\n\n"
        "• Обычный - использует только модель для генерации ответа\n"
        "• Умный - дополнительно использует базу знаний для более точного ответа",
        reply_markup=keyboard
    )

@router.callback_query(F.data.in_(["regular_inference", "rag_inference"]))
async def handle_inference_type_choice(callback: CallbackQuery, state: FSMContext):
    """Handle inference type selection"""
    await callback.answer()
    
    # Store inference type in state
    use_rag = callback.data == "rag_inference"
    await state.update_data(use_rag=use_rag)
    await state.set_state(UnifiedInferenceState.waiting_for_prompt)
    
    type_text = "умного" if use_rag else "обычного"
    await callback.message.answer(
        f"✍️ Режим {type_text} запроса.\n\n"
        "Пожалуйста, напишите ваш вопрос о стиле или моде."
    )

@router.message(UnifiedInferenceState.waiting_for_prompt)
async def handle_unified_prompt(message: Message, state: FSMContext):
    """Handle user's prompt for unified inference"""
    try:
        # Get inference type from state
        state_data = await state.get_data()
        use_rag = state_data.get("use_rag", False)
        
        await message.answer("⏳ Обрабатываю ваш запрос...")
        
        # Prepare request data
        request_data = {
            "prompt": message.text,
            "inference_type": "rag" if use_rag else "direct",
            "model_params": {
                "temperature": 0.7,
                "max_length": 500,
                "top_p": 0.9
            }
        }
        
        # Add RAG params if using RAG
        if use_rag:
            request_data["rag_params"] = {
                "search_depth": 2,
                "max_context_chunks": 5,
                "similarity_threshold": 0.7
            }
        
        # Make request to API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/inference",
                json=request_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    task_id = result.get("task_id")
                    
                    if task_id:
                        # Poll for result
                        for _ in range(30):  # 30 seconds timeout
                            async with session.get(
                                f"{API_BASE_URL}/status/{task_id}"
                            ) as status_response:
                                if status_response.status == 200:
                                    status_data = await status_response.json()
                                    if status_data["status"] == "completed":
                                        # Format and send response
                                        await format_and_send_response(
                                            message,
                                            status_data["result"],
                                            use_rag
                                        )
                                        break
                            await asyncio.sleep(1)
                        else:
                            await message.answer(
                                "⚠️ Запрос занял слишком много времени.\n"
                                "Пожалуйста, попробуйте еще раз или измените запрос."
                            )
                    else:
                        await message.answer(
                            "❌ Не удалось получить идентификатор задачи.\n"
                            "Пожалуйста, попробуйте позже."
                        )
                else:
                    error_data = await response.json()
                    logger.error(f"API error: {error_data}")
                    await message.answer(
                        "❌ Произошла ошибка при обработке запроса.\n"
                        "Пожалуйста, попробуйте позже."
                    )
    
    except Exception as e:
        logger.error(f"Error in unified inference handler: {e}")
        await message.answer(
            "❌ Произошла ошибка при обработке запроса.\n"
            "Пожалуйста, попробуйте позже."
        )
    finally:
        await state.clear()

async def format_and_send_response(message: Message, result: Dict[str, Any], use_rag: bool):
    """Format and send the inference response"""
    response_parts = []
    
    # Add main response
    if isinstance(result, str):
        response_parts.append(result)
    elif isinstance(result, dict):
        response_parts.append(result.get("generated_text", ""))
        
        # Add RAG info if available and using RAG
        if use_rag and result.get("rag_info"):
            rag_info = result["rag_info"]
            if rag_info.get("extracted_nodes"):
                response_parts.append("\n🔍 Использованная информация из базы знаний:")
                for node in rag_info["extracted_nodes"][:3]:  # Show top 3 nodes
                    response_parts.append(f"• {node['name']}")
            
            if rag_info.get("prompt_additions"):
                response_parts.append("\n💡 Дополнительный контекст:")
                for addition in rag_info["prompt_additions"][:3]:  # Show top 3 additions
                    response_parts.append(f"• {addition}")
        
        # Add processing time if available
        if result.get("processing_time"):
            response_parts.append(
                f"\n⏱ Время обработки: {result['processing_time']:.2f} сек."
            )
    
    # Send response
    await message.answer("\n".join(response_parts)) 