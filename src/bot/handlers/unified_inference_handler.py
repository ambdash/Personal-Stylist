from aiogram import Router, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from src.bot.states import UnifiedInferenceState
from src.bot.keyboards import get_inference_type_keyboard
from src.celery_app import app as celery_app
import logging
import json
from typing import Dict, Any
import asyncio

logger = logging.getLogger(__name__)
router = Router()

# Default parameters
DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "max_new_tokens": 512,
    "do_sample": True,
    "num_beams": 1
}

@router.message(Command("ask"))
async def cmd_unified_inference(message: Message, state: FSMContext):
    """Handle /ask command"""
    keyboard = get_inference_type_keyboard()
    await state.set_state(UnifiedInferenceState.choosing_type)
    await message.answer(
        "🤖 Выберите режим запроса:\n\n"
        "• Обычный - использует только модель для генерации ответа\n"
        "• Умный - дополнительно использует базу знаний для более точного ответа\n\n"
        "После выбора режима вы сможете настроить параметры генерации.",
        reply_markup=keyboard
    )

@router.message(Command("ask_with_params"))
async def cmd_inference_with_params(message: Message, state: FSMContext):
    """Handle /ask_with_params command for advanced users"""
    # Parse parameters from command
    parts = message.text.split()
    if len(parts) < 2:
        await message.answer(
            "❌ Неверный формат команды.\n\n"
            "Использование: /ask_with_params <prompt> [параметры]\n\n"
            "Пример: /ask_with_params Как носить джинсы? temperature=0.8 use_rag=true\n\n"
            "Доступные параметры:\n"
            "• temperature (0.1-2.0) - креативность\n"
            "• top_p (0.1-1.0) - разнообразие\n"
            "• top_k (1-100) - количество вариантов\n"
            "• repetition_penalty (1.0-2.0) - избежание повторов\n"
            "• max_new_tokens (50-1000) - длина ответа\n"
            "• use_rag (true/false) - использовать базу знаний"
        )
        return
    
    # Extract prompt and parameters
    prompt_and_params = " ".join(parts[1:])
    prompt_parts = prompt_and_params.split()
    
    # Find where parameters start (look for key=value pattern)
    prompt_end = len(prompt_parts)
    for i, part in enumerate(prompt_parts):
        if "=" in part:
            prompt_end = i
            break
    
    prompt = " ".join(prompt_parts[:prompt_end])
    param_parts = prompt_parts[prompt_end:]
    
    # Parse parameters
    params = DEFAULT_PARAMS.copy()
    use_rag = False
    
    for param in param_parts:
        if "=" in param:
            key, value = param.split("=", 1)
            if key == "use_rag":
                use_rag = value.lower() in ["true", "1", "yes"]
            elif key == "temperature":
                try:
                    params["temperature"] = max(0.1, min(2.0, float(value)))
                except ValueError:
                    pass
            elif key == "top_p":
                try:
                    params["top_p"] = max(0.1, min(1.0, float(value)))
                except ValueError:
                    pass
            elif key == "top_k":
                try:
                    params["top_k"] = max(1, min(100, int(value)))
                except ValueError:
                    pass
            elif key == "repetition_penalty":
                try:
                    params["repetition_penalty"] = max(1.0, min(2.0, float(value)))
                except ValueError:
                    pass
            elif key == "max_new_tokens":
                try:
                    params["max_new_tokens"] = max(50, min(1000, int(value)))
                except ValueError:
                    pass
    
    if not prompt.strip():
        await message.answer("❌ Пожалуйста, укажите вопрос.")
        return
    
    # Process the request
    await process_inference_request(message, prompt, use_rag, params)

@router.callback_query(F.data.in_(["regular_inference", "rag_inference"]))
async def handle_inference_type_choice(callback: CallbackQuery, state: FSMContext):
    """Handle inference type selection"""
    await callback.answer()
    
    # Store inference type in state
    use_rag = callback.data == "rag_inference"
    await state.update_data(use_rag=use_rag, parameters=DEFAULT_PARAMS.copy())
    
    # Show parameter configuration options
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎯 Использовать стандартные параметры", callback_data="use_default_params")],
        [InlineKeyboardButton(text="⚙️ Настроить параметры", callback_data="configure_params")],
        [InlineKeyboardButton(text="ℹ️ Что означают параметры?", callback_data="explain_params")]
    ])
    
    type_text = "умного" if use_rag else "обычного"
    await callback.message.answer(
        f"✅ Выбран режим {type_text} запроса.\n\n"
        "Теперь выберите настройки генерации:",
        reply_markup=keyboard
    )

@router.callback_query(F.data == "use_default_params")
async def use_default_params(callback: CallbackQuery, state: FSMContext):
    """Use default parameters"""
    await callback.answer()
    await state.set_state(UnifiedInferenceState.waiting_for_prompt)
    
    await callback.message.answer(
        "✍️ Стандартные параметры установлены.\n\n"
        "Пожалуйста, напишите ваш вопрос о стиле или моде."
    )

@router.callback_query(F.data == "configure_params")
async def configure_params(callback: CallbackQuery, state: FSMContext):
    """Configure parameters"""
    await callback.answer()
    
    state_data = await state.get_data()
    params = state_data.get("parameters", DEFAULT_PARAMS.copy())
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"🌡️ Креативность: {params['temperature']}", callback_data="param_temperature")],
        [InlineKeyboardButton(text=f"🎲 Разнообразие: {params['top_p']}", callback_data="param_top_p")],
        [InlineKeyboardButton(text=f"📊 Варианты: {params['top_k']}", callback_data="param_top_k")],
        [InlineKeyboardButton(text=f"🔄 Повторы: {params['repetition_penalty']}", callback_data="param_repetition")],
        [InlineKeyboardButton(text=f"📏 Длина: {params['max_new_tokens']}", callback_data="param_length")],
        [InlineKeyboardButton(text="✅ Готово", callback_data="params_done")]
    ])
    
    await callback.message.answer(
        "⚙️ Настройка параметров генерации:\n\n"
        f"🌡️ Креативность (temperature): {params['temperature']}\n"
        f"🎲 Разнообразие (top_p): {params['top_p']}\n"
        f"📊 Количество вариантов (top_k): {params['top_k']}\n"
        f"🔄 Избежание повторов: {params['repetition_penalty']}\n"
        f"📏 Максимальная длина: {params['max_new_tokens']} токенов\n\n"
        "Нажмите на параметр для изменения:",
        reply_markup=keyboard
    )

@router.callback_query(F.data == "explain_params")
async def explain_params(callback: CallbackQuery):
    """Explain what parameters mean"""
    await callback.answer()
    
    explanation = (
        "📚 Объяснение параметров:\n\n"
        "🌡️ **Креативность (Temperature)**\n"
        "0.1-0.5: Более предсказуемые ответы\n"
        "0.6-0.8: Сбалансированные ответы\n"
        "0.9-2.0: Более творческие ответы\n\n"
        "🎲 **Разнообразие (Top-p)**\n"
        "0.1-0.5: Консервативный выбор слов\n"
        "0.6-0.9: Сбалансированный выбор\n"
        "0.9-1.0: Максимальное разнообразие\n\n"
        "📊 **Количество вариантов (Top-k)**\n"
        "1-20: Ограниченный выбор\n"
        "21-50: Средний выбор\n"
        "51-100: Широкий выбор\n\n"
        "🔄 **Избежание повторов**\n"
        "1.0: Без штрафа за повторы\n"
        "1.1-1.3: Легкий штраф\n"
        "1.4-2.0: Сильный штраф\n\n"
        "📏 **Длина ответа**\n"
        "50-200: Короткий ответ\n"
        "201-500: Средний ответ\n"
        "501-1000: Длинный ответ"
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚙️ Настроить параметры", callback_data="configure_params")],
        [InlineKeyboardButton(text="🎯 Использовать стандартные", callback_data="use_default_params")]
    ])
    
    await callback.message.answer(explanation, reply_markup=keyboard, parse_mode="Markdown")

@router.callback_query(F.data == "params_done")
async def params_done(callback: CallbackQuery, state: FSMContext):
    """Finish parameter configuration"""
    await callback.answer()
    await state.set_state(UnifiedInferenceState.waiting_for_prompt)
    
    await callback.message.answer(
        "✅ Параметры настроены!\n\n"
        "Теперь напишите ваш вопрос о стиле или моде."
    )

@router.message(UnifiedInferenceState.waiting_for_prompt)
async def handle_unified_prompt(message: Message, state: FSMContext):
    """Handle user's prompt for unified inference"""
    try:
        # Get inference type and parameters from state
        state_data = await state.get_data()
        use_rag = state_data.get("use_rag", False)
        params = state_data.get("parameters", DEFAULT_PARAMS.copy())
        
        await process_inference_request(message, message.text, use_rag, params)
        
    except Exception as e:
        logger.error(f"Error in unified inference handler: {e}")
        await message.answer(
            "❌ Произошла ошибка при обработке запроса.\n"
            "Пожалуйста, попробуйте позже."
        )
    finally:
        await state.clear()

async def process_inference_request(message: Message, prompt: str, use_rag: bool, params: Dict[str, Any]):
    """Process inference request using Celery tasks"""
    processing_msg = await message.answer("⏳ Обрабатываю ваш запрос...")
    
    try:
        # Submit task to Celery worker
        task = celery_app.send_task(
            'inference_worker.generate_text',
            kwargs={
                'prompt': prompt,
                'use_rag': use_rag,
                'model_name': "t-tech/T-lite-it-1.0",
                'adapter_path': None,  # Use default fine-tuned adapter
                'system_prompt': None,
                'parameters': params
            },
            queue='inference'
        )
        
        # Wait for task completion with periodic status updates
        max_wait_time = 300  # 5 minutes
        check_interval = 10  # Check every 10 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                # Check if task is ready
                if task.ready():
                    result = task.result
                    break
                
                # Update status every 30 seconds
                if elapsed_time > 0 and elapsed_time % 30 == 0:
                    await processing_msg.edit_text(
                        f"⏳ Обрабатываю ваш запрос... ({elapsed_time}с)"
                    )
                
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
                
            except Exception as e:
                logger.error(f"Error checking task status: {e}")
                break
        else:
            # Timeout reached
            await processing_msg.edit_text(
                "⚠️ Запрос занял слишком много времени.\n"
                "Пожалуйста, попробуйте еще раз или измените запрос."
            )
            return
        
        # Process result
        if result and result.get('success'):
            task_result = result['result']
            await format_and_send_response(message, processing_msg, task_result, use_rag, params)
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Task failed'
            await processing_msg.edit_text(
                f"❌ Произошла ошибка при обработке запроса: {error_msg}\n"
                "Пожалуйста, попробуйте позже."
            )
    
    except Exception as e:
        logger.error(f"Celery task error: {e}")
        await processing_msg.edit_text(
            "❌ Произошла ошибка при отправке запроса.\n"
            "Пожалуйста, попробуйте позже."
        )

async def format_and_send_response(
    message: Message, 
    processing_msg: Message, 
    result: Dict[str, Any], 
    use_rag: bool, 
    params: Dict[str, Any]
):
    """Format and send the inference response"""
    response_parts = []
    
    # Add main response
    response_parts.append(result.get("generated_text", ""))
    
    # Add RAG info if available and using RAG
    if use_rag and result.get("rag_info"):
        rag_info = result["rag_info"]
        if rag_info.get("enhanced"):
            response_parts.append("\n🔍 Использована информация из базы знаний")
        else:
            response_parts.append("\n💡 База знаний не содержит релевантной информации")
    
    # Add technical info
    tech_info = []
    tech_info.append(f"⏱ Время: {result.get('processing_time', 0):.2f}с")
    tech_info.append(f"🤖 Модель: {result.get('model_used', 'unknown')}")
    
    # Add parameter info if different from defaults
    used_params = result.get("parameters", {})
    param_info = []
    if used_params.get("temperature", DEFAULT_PARAMS["temperature"]) != DEFAULT_PARAMS["temperature"]:
        param_info.append(f"🌡️{used_params['temperature']}")
    if used_params.get("top_p", DEFAULT_PARAMS["top_p"]) != DEFAULT_PARAMS["top_p"]:
        param_info.append(f"🎲{used_params['top_p']}")
    if used_params.get("top_k", DEFAULT_PARAMS["top_k"]) != DEFAULT_PARAMS["top_k"]:
        param_info.append(f"📊{used_params['top_k']}")
    
    if param_info:
        tech_info.append(f"⚙️ {' '.join(param_info)}")
    
    response_parts.append(f"\n{' | '.join(tech_info)}")
    
    # Send response (delete processing message and send new one)
    await processing_msg.delete()
    await message.answer("\n".join(response_parts)) 