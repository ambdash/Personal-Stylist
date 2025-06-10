import asyncio
import time
import aiohttp
import os
from typing import Dict, Any
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from src.bot.states import UnifiedInferenceState
from src.bot.keyboards import get_inference_type_keyboard
from src.api.services.enhanced_rag_service import EnhancedRagService
from src.api.db.neo4j.config import Neo4jConnection
import logging
import json

logger = logging.getLogger(__name__)
router = Router()

# Get API base URL from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize RAG service
rag_service = None

def get_rag_service():
    """Get or initialize RAG service"""
    global rag_service
    if rag_service is None:
        try:
            rag_service = EnhancedRagService()
            logger.info("Enhanced RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            rag_service = None
    return rag_service

# Default parameters
DEFAULT_PARAMS = {
    "temperature": 0.5,  # Less creative, more consistent
    "top_p": 0.8,        # More focused word selection
    "top_k": 30,         # Fewer word options
    "repetition_penalty": 1.1,  # Less aggressive repetition penalty
    "max_new_tokens": 256,      # Shorter responses
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
    
    logger.info(f"User selected inference type: {callback.data}")
    
    # Store inference type in state
    use_rag = callback.data == "rag_inference"
    await state.update_data(use_rag=use_rag, parameters=DEFAULT_PARAMS.copy())
    
    logger.info(f"Updated state: use_rag={use_rag}")
    
    # Show parameter configuration options
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🎯 Использовать стандартные параметры", callback_data="use_default_params")],
        [InlineKeyboardButton(text="⚙️ Настроить параметры", callback_data="configure_params")],
        [InlineKeyboardButton(text="ℹ️ Что означают параметры?", callback_data="explain_params")]
    ])
    
    type_text = "умного" if use_rag else "обычного"
    # Edit the existing message instead of sending a new one
    await callback.message.edit_text(
        f"✅ Выбран режим {type_text} запроса.\n\n"
        "Теперь выберите настройки генерации:",
        reply_markup=keyboard
    )

@router.callback_query(F.data == "use_default_params")
async def use_default_params(callback: CallbackQuery, state: FSMContext):
    """Use default parameters"""
    await callback.answer()
    
    logger.info("User selected default parameters")
    
    await state.set_state(UnifiedInferenceState.waiting_for_prompt)
    
    logger.info("State set to waiting_for_prompt")
    
    # Edit the existing message instead of sending a new one
    await callback.message.edit_text(
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
    
    # Edit the existing message instead of sending a new one
    await callback.message.edit_text(
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
    
    # Edit the existing message instead of sending a new one
    await callback.message.edit_text(explanation, reply_markup=keyboard, parse_mode="Markdown")

@router.callback_query(F.data == "params_done")
async def params_done(callback: CallbackQuery, state: FSMContext):
    """Finish parameter configuration"""
    await callback.answer()
    await state.set_state(UnifiedInferenceState.waiting_for_prompt)
    
    # Edit the existing message instead of sending a new one
    await callback.message.edit_text(
        "✅ Параметры настроены!\n\n"
        "Теперь напишите ваш вопрос о стиле или моде."
    )

@router.message(UnifiedInferenceState.waiting_for_prompt)
async def handle_unified_prompt(message: Message, state: FSMContext):
    """Handle user's prompt for unified inference"""
    try:
        logger.info(f"Received message in waiting_for_prompt state: {message.text[:50]}...")
        
        # Get inference type and parameters from state
        state_data = await state.get_data()
        use_rag = state_data.get("use_rag", False)
        params = state_data.get("parameters", DEFAULT_PARAMS.copy())
        
        logger.info(f"State data: use_rag={use_rag}, params={params}")
        
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
    """Process inference request with enhanced RAG integration"""
    # Send initial status message
    processing_msg = await message.answer("🤖 Начинаю обработку запроса...")
    start_time = time.time()
    
    try:
        # If RAG is enabled, first process with RAG service
        rag_info = None
        enhanced_prompt = prompt
        auxiliary_info_msg = None
        
        if use_rag:
            try:
                # Update status
                await processing_msg.edit_text("🔍 Анализирую запрос и ищу релевантную информацию...")
                
                # Get RAG service
                rag_svc = get_rag_service()
                if rag_svc is None:
                    await processing_msg.edit_text("❌ Сервис базы знаний недоступен. Переключаюсь на обычный режим...")
                    use_rag = False
                else:
                    # Process with RAG
                    rag_result = await rag_svc.enhance_prompt_async(prompt)
                    
                    if rag_result and rag_result.get("enhanced", False):
                        enhanced_prompt = rag_result["enhanced_prompt"]
                        rag_info = rag_result
                        
                        # Show auxiliary information to user
                        aux_info = format_auxiliary_info(rag_result)
                        auxiliary_info_msg = await message.answer(aux_info, parse_mode="Markdown")
                        
                        # Update processing message
                        await processing_msg.edit_text("🧠 Генерирую ответ с использованием найденной информации...")
                    else:
                        await processing_msg.edit_text("💡 Релевантная информация не найдена. Генерирую ответ на основе модели...")
                        use_rag = False
                        
            except Exception as e:
                logger.error(f"RAG processing error: {e}")
                await processing_msg.edit_text("⚠️ Ошибка при работе с базой знаний. Переключаюсь на обычный режим...")
                use_rag = False
        else:
            await processing_msg.edit_text("🤖 Генерирую ответ...")
        
        # Prepare request data for API
        request_data = {
            "prompt": enhanced_prompt,
            "use_rag": False,  # We already processed RAG, so send enhanced prompt directly
            "parameters": params
        }
        
        # Make API call to unified inference
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/v1/unified_inference/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    total_time = time.time() - start_time
                    result['total_time'] = total_time
                    
                    # Add RAG info to result if available
                    if rag_info:
                        result['rag_info'] = rag_info
                    
                    await format_and_send_response(message, processing_msg, result, use_rag, params, auxiliary_info_msg)
                else:
                    error_text = await response.text()
                    total_time = time.time() - start_time
                    await processing_msg.edit_text(
                        f"❌ Ошибка API (код {response.status}): {error_text}\n"
                        f"⏱ Время выполнения: {total_time:.1f}с"
                    )
    
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        await processing_msg.edit_text(
            f"⚠️ Запрос занял слишком много времени ({total_time:.1f}с).\n"
            "Пожалуйста, попробуйте еще раз или измените запрос."
        )
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"API request error: {e}")
        await processing_msg.edit_text(
            f"❌ Произошла ошибка при отправке запроса: {str(e)}\n"
            f"⏱ Время выполнения: {total_time:.1f}с\n"
            "Пожалуйста, попробуйте позже."
        )

def format_auxiliary_info(rag_result: Dict[str, Any]) -> str:
    """Format auxiliary information message for user"""
    info_parts = ["📚 **Вспомогательная информация:**\n"]
    
    # Strategy used
    strategy = rag_result.get("strategy_used", "неизвестно")
    strategy_names = {
        "item_type_strategy": "🎯 Поиск по типу предмета",
        "key_node_intersection": "🔗 Пересечение ключевых понятий", 
        "single_node_expansion": "📈 Расширение одного понятия",
        "fallback_strategy": "🔄 Резервная стратегия"
    }
    strategy_name = strategy_names.get(strategy, strategy)
    info_parts.append(f"**Стратегия:** {strategy_name}")
    
    # Item type detection
    item_type = rag_result.get("item_type", "")
    item_subtype = rag_result.get("item_subtype", "")
    if item_type:
        item_info = f"**Тип предмета:** {item_type}"
        if item_subtype and item_subtype != item_type:
            item_info += f" → {item_subtype}"
        info_parts.append(item_info)
    
    # Styling context
    is_styling = rag_result.get("is_styling", False)
    styling_item = rag_result.get("styling_item", "")
    if is_styling:
        styling_info = "**Контекст стилизации:** Да"
        if styling_item:
            styling_info += f" ({styling_item})"
        info_parts.append(styling_info)
    
    # Key nodes found - detailed breakdown by type
    key_nodes = rag_result.get("key_nodes_found", [])
    if key_nodes:
        # Group by type
        nodes_by_type = {}
        for node in key_nodes:
            node_type = node.get("type", "Неизвестно")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node.get("name", "неизвестно"))
        
        info_parts.append("**Найденные ключевые понятия:**")
        for node_type, names in nodes_by_type.items():
            # Add emoji for each type
            type_emojis = {
                "Сезон": "🌸",
                "Погода": "🌤️", 
                "Случай": "🎭",
                "Эстетика": "✨",
                "Тренд": "📈"
            }
            emoji = type_emojis.get(node_type, "🔹")
            info_parts.append(f"  {emoji} {node_type}: {', '.join(names)}")
    
    # Concepts used with intersection scores
    concepts_used = rag_result.get("concepts_used", [])
    if concepts_used:
        concept_count = len(concepts_used)
        info_parts.append(f"**Найдено концепций:** {concept_count}")
        
        # Show top concepts with their intersection scores
        for i, concept in enumerate(concepts_used[:3], 1):
            name = concept.get("name", "неизвестно")
            if ':' in name:
                name = name.split(':', 1)[1]  # Remove type prefix
            
            intersection_score = concept.get("intersection_score", 0)
            relations_count = len(concept.get("relations", []))
            
            score_emoji = "🎯" if intersection_score >= 2 else "🔸" if intersection_score >= 1 else "🔹"
            info_parts.append(f"  {i}. {score_emoji} {name} (пересечений: {intersection_score}, связей: {relations_count})")
        
        if len(concepts_used) > 3:
            info_parts.append(f"  *(и еще {len(concepts_used) - 3})*")
    
    # Processing time
    processing_time = rag_result.get("processing_time", 0)
    info_parts.append(f"**Время поиска:** {processing_time:.2f}с")
    
    info_parts.append("\n🔄 *Генерирую ответ с учетом найденной информации...*")
    
    return "\n".join(info_parts)

async def format_and_send_response(
    message: Message, 
    processing_msg: Message, 
    result: Dict[str, Any], 
    use_rag: bool, 
    params: Dict[str, Any],
    auxiliary_info_msg: Message = None
):
    """Format and send the inference response"""
    response_parts = []
    
    # Add main response
    response_parts.append(result.get("generated_text", ""))
    
    # Add RAG info if available and using RAG
    if use_rag and result.get("rag_info"):
        rag_info = result["rag_info"]
        if rag_info.get("enhanced"):
            concepts_count = len(rag_info.get("concepts_used", []))
            response_parts.append(f"\n✅ *Использована информация из базы знаний ({concepts_count} концепций)*")
        else:
            response_parts.append("\n💡 *База знаний не содержит релевантной информации*")
    
    # Add technical info - only inference time
    tech_info = []
    
    # Use total time if available, otherwise processing time
    total_time = result.get('total_time', result.get('processing_time', 0))
    tech_info.append(f"⏱ Время генерации: {total_time:.1f}с")
    
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
    
    # Send the main response
    await message.answer("\n".join(response_parts), parse_mode="Markdown")
    
    # Update auxiliary info message if it exists
    if auxiliary_info_msg and use_rag:
        try:
            await auxiliary_info_msg.edit_text(
                auxiliary_info_msg.text.replace(
                    "🔄 *Генерирую ответ с учетом найденной информации...*",
                    "✅ *Ответ сгенерирован с использованием найденной информации*"
                ),
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.warning(f"Failed to update auxiliary info message: {e}")

@router.message(Command("debug_rag"))
async def cmd_debug_rag(message: Message):
    """Handle /debug_rag command for testing RAG service"""
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer(
            "🔍 **Отладка RAG сервиса**\n\n"
            "Использование: `/debug_rag <ваш вопрос>`\n\n"
            "Этот режим покажет детальную информацию о том, как RAG сервис обрабатывает ваш запрос:\n"
            "• Какие ключевые понятия найдены\n"
            "• Какая стратегия поиска используется\n"
            "• Какие концепции найдены и их оценки\n"
            "• Время обработки\n\n"
            "Пример: `/debug_rag Какую обувь носить летом в офис в 2025?`",
            parse_mode="Markdown"
        )
        return
    
    prompt = parts[1].strip()
    
    # Send initial status message
    processing_msg = await message.answer("🔍 Анализирую запрос через RAG сервис...")
    start_time = time.time()
    
    try:
        # Get RAG service
        rag_svc = get_rag_service()
        if rag_svc is None:
            await processing_msg.edit_text("❌ RAG сервис недоступен")
            return
        
        # Process with RAG
        rag_result = await rag_svc.enhance_prompt_async(prompt)
        total_time = time.time() - start_time
        
        if not rag_result:
            await processing_msg.edit_text("❌ Ошибка при обработке запроса")
            return
        
        # Format detailed debug information
        debug_info = format_debug_info(rag_result, total_time)
        
        # Delete processing message and send debug info
        await processing_msg.delete()
        await message.answer(debug_info, parse_mode="Markdown")
        
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Debug RAG error: {e}")
        await processing_msg.edit_text(
            f"❌ Ошибка при отладке RAG: {str(e)}\n"
            f"⏱ Время: {total_time:.2f}с"
        )

def format_debug_info(rag_result: Dict[str, Any], total_time: float) -> str:
    """Format detailed debug information for RAG processing"""
    info_parts = ["🔍 **Детальная отладка RAG сервиса**\n"]
    
    # Original vs Enhanced prompt
    original = rag_result.get("original_prompt", "")
    enhanced = rag_result.get("enhanced_prompt", "")
    info_parts.append(f"**Исходный запрос:** {original}")
    
    if enhanced != original:
        # Show only the added part
        added_info = enhanced.replace(original, "").strip()
        if added_info:
            info_parts.append(f"**Добавленная информация:**\n```\n{added_info}\n```")
    else:
        info_parts.append("**Результат:** Запрос не был дополнен")
    
    # Status and strategy
    status = rag_result.get("status", "неизвестно")
    strategy = rag_result.get("strategy_used", "неизвестно")
    enhanced_flag = rag_result.get("enhanced", False)
    
    info_parts.append(f"**Статус:** {status}")
    info_parts.append(f"**Дополнен:** {'✅ Да' if enhanced_flag else '❌ Нет'}")
    
    strategy_names = {
        "item_type_strategy": "🎯 Поиск по типу предмета",
        "key_node_intersection": "🔗 Пересечение ключевых понятий", 
        "single_node_expansion": "📈 Расширение одного понятия",
        "fallback_strategy": "🔄 Резервная стратегия"
    }
    strategy_name = strategy_names.get(strategy, strategy)
    info_parts.append(f"**Стратегия:** {strategy_name}")
    
    # Item detection
    item_type = rag_result.get("item_type", "")
    item_subtype = rag_result.get("item_subtype", "")
    is_styling = rag_result.get("is_styling", False)
    styling_item = rag_result.get("styling_item", "")
    
    if item_type or is_styling:
        info_parts.append("\n**🎯 Анализ предметов:**")
        if item_type:
            item_info = f"• Тип: {item_type}"
            if item_subtype and item_subtype != item_type:
                item_info += f" → {item_subtype}"
            info_parts.append(item_info)
        
        if is_styling:
            styling_info = "• Контекст стилизации: Да"
            if styling_item:
                styling_info += f" ({styling_item})"
            info_parts.append(styling_info)
    
    # Key nodes analysis
    key_nodes = rag_result.get("key_nodes_found", [])
    if key_nodes:
        info_parts.append("\n**🔑 Ключевые понятия:**")
        
        # Group by type
        nodes_by_type = {}
        for node in key_nodes:
            node_type = node.get("type", "Неизвестно")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append({
                "name": node.get("name", "неизвестно"),
                "id": node.get("id", ""),
                "score": node.get("score", 0.0)
            })
        
        for node_type, nodes in nodes_by_type.items():
            type_emojis = {
                "Сезон": "🌸",
                "Погода": "🌤️", 
                "Случай": "🎭",
                "Эстетика": "✨",
                "Тренд": "📈"
            }
            emoji = type_emojis.get(node_type, "🔹")
            info_parts.append(f"**{emoji} {node_type}:**")
            
            for node in nodes:
                score_text = f" (оценка: {node['score']})" if node['score'] > 0 else ""
                info_parts.append(f"  • {node['name']}{score_text}")
    else:
        info_parts.append("\n**🔑 Ключевые понятия:** Не найдены")
    
    # Concepts analysis
    concepts_used = rag_result.get("concepts_used", [])
    if concepts_used:
        info_parts.append(f"\n**💡 Найденные концепции ({len(concepts_used)}):**")
        
        for i, concept in enumerate(concepts_used[:5], 1):  # Show top 5
            name = concept.get("name", "неизвестно")
            if ':' in name:
                name = name.split(':', 1)[1]  # Remove type prefix
            
            intersection_score = concept.get("intersection_score", 0)
            relations_count = len(concept.get("relations", []))
            
            score_emoji = "🎯" if intersection_score >= 2 else "🔸" if intersection_score >= 1 else "🔹"
            info_parts.append(f"{i}. {score_emoji} **{name}**")
            info_parts.append(f"   • Пересечений с ключевыми понятиями: {intersection_score}")
            info_parts.append(f"   • Всего связей: {relations_count}")
            
            # Show some relations
            relations = concept.get("relations", [])
            if relations:
                rel_examples = []
                for rel in relations[:3]:  # Show max 3 relations
                    rel_type = rel.get("type", "")
                    node = rel.get("node", {})
                    if node and node.get("name"):
                        rel_examples.append(f"{rel_type}: {node['name']}")
                
                if rel_examples:
                    info_parts.append(f"   • Примеры связей: {', '.join(rel_examples)}")
        
        if len(concepts_used) > 5:
            info_parts.append(f"\n*(и еще {len(concepts_used) - 5} концепций)*")
    else:
        info_parts.append("\n**💡 Концепции:** Не найдены")
    
    # Timing information
    processing_time = rag_result.get("processing_time", 0)
    info_parts.append(f"\n**⏱ Время обработки:** {processing_time:.3f}с")
    info_parts.append(f"**⏱ Общее время:** {total_time:.3f}с")
    
    return "\n".join(info_parts) 