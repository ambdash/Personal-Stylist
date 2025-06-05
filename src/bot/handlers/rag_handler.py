from aiogram import Router, F
from aiogram.types import Message
from aiogram.filters import Command
from src.api.services.rag_service import RagService
from src.api.db.neo4j.config import Neo4jConnection
from src.bot.states import RagState
from aiogram.fsm.context import FSMContext
import logging

logger = logging.getLogger(__name__)
router = Router()

def get_neo4j_connection():
    """Create Neo4j connection"""
    return Neo4jConnection()

@router.message(Command("rag"))
async def start_rag(message: Message, state: FSMContext):
    """Start RAG interaction"""
    await message.answer(
        "Пожалуйста, отправьте ваш вопрос о моде и стиле. "
        "Я проанализирую его с помощью базы знаний и предоставлю релевантную информацию."
    )
    await state.set_state(RagState.waiting_for_prompt)

@router.message(RagState.waiting_for_prompt)
async def process_rag_query(message: Message, state: FSMContext):
    """Process the user's fashion query through RAG pipeline"""
    try:
        # Initialize RAG service
        rag_service = RagService(get_neo4j_connection())
        
        # Process the query
        result = await rag_service.process_rag_query(message.text)
        
        if result["status"] == "no_nodes_found":
            await message.answer(
                "К сожалению, я не смог найти в вашем запросе терминов, связанных с модой. "
                "Пожалуйста, попробуйте переформулировать запрос, используя более конкретные термины из мира моды."
            )
            return

        # Format the response
        response_text = "🔍 Результаты анализа:\n\n"
        
        if result["extracted_nodes"]:
            response_text += "📌 Найденные термины:\n"
            for node in result["extracted_nodes"]:
                response_text += f"• {node['label']}: {node['name']}\n"
            response_text += "\n"
        
        if result["knowledge_chunks"]:
            response_text += "🔗 Найденные связи в базе знаний:\n\n"
            for chunk in result["knowledge_chunks"]:
                chunk_texts = []
                for path in chunk:
                    formatted = rag_service.format_path_for_prompt(path)
                    if formatted:
                        chunk_texts.append(formatted)
                if chunk_texts:
                    response_text += "• " + "\n• ".join(chunk_texts) + "\n\n"
        
        # Add prompt enhancement suggestion
        if result["prompt_additions"]:
            response_text += "\n💡 Рекомендация по использованию этой информации:\n"
            response_text += "Вы можете дополнить ваш запрос следующей информацией:\n\n"
            for i, addition in enumerate(result["prompt_additions"], 1):
                response_text += f"{i}. {addition}\n"
        
        await message.answer(response_text)
        await state.clear()
        
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        await message.answer(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, попробуйте позже."
        )
        await state.clear() 