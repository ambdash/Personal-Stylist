#!/usr/bin/env python3
"""
Test script to demonstrate enhanced RAG integration in Telegram bot
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.services.enhanced_rag_service import EnhancedRagService
from src.api.db.neo4j.config import Neo4jConnection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_auxiliary_info(rag_result):
    """Format auxiliary information message for user (same as in bot)"""
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
    
    # Key nodes found
    key_nodes = rag_result.get("key_nodes_found", [])
    if key_nodes:
        node_names = [node.get("name", "неизвестно") for node in key_nodes[:3]]  # Show max 3
        info_parts.append(f"**Ключевые понятия:** {', '.join(node_names)}")
        if len(key_nodes) > 3:
            info_parts.append(f"*(и еще {len(key_nodes) - 3})*")
    
    # Concepts used
    concepts_used = rag_result.get("concepts_used", [])
    if concepts_used:
        concept_count = len(concepts_used)
        concept_names = []
        for concept in concepts_used[:2]:  # Show max 2 concept names
            name = concept.get("name", "неизвестно")
            relations_count = len(concept.get("relations", []))
            concept_names.append(f"{name} ({relations_count} связей)")
        
        info_parts.append(f"**Найдено концепций:** {concept_count}")
        if concept_names:
            info_parts.append(f"**Примеры:** {', '.join(concept_names)}")
    
    # Processing time
    processing_time = rag_result.get("processing_time", 0)
    info_parts.append(f"**Время поиска:** {processing_time:.2f}с")
    
    info_parts.append("\n🔄 *Генерирую ответ с учетом найденной информации...*")
    
    return "\n".join(info_parts)

async def test_bot_rag_integration():
    """Test the RAG integration as it would work in the bot"""
    
    print("🤖 Тестирование интеграции RAG в Telegram бот\n")
    
    # Initialize RAG service
    try:
        rag_service = EnhancedRagService()
        print("✅ Enhanced RAG service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG service: {e}")
        return
    
    # Test scenarios
    test_prompts = [
        "Как стилизовать платье для офиса зимой?",
        "Какие туфли подойдут к джинсам и свитеру?",
        "Подскажи образ для свидания в ресторане",
        "Что носить в минималистичном стиле?",
        "Как сочетать черный и белый цвета?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Тест {i}: {prompt}")
        print('='*60)
        
        # Simulate bot processing
        print("🤖 Начинаю обработку запроса...")
        print("🔍 Анализирую запрос и ищу релевантную информацию...")
        
        try:
            # Process with RAG
            rag_result = await rag_service.enhance_prompt_async(prompt)
            
            if rag_result and rag_result.get("enhanced", False):
                # Show auxiliary information (as bot would show to user)
                aux_info = format_auxiliary_info(rag_result)
                print("\n" + aux_info.replace("**", "").replace("*", ""))
                
                print("\n🧠 Генерирую ответ с использованием найденной информации...")
                
                # Show what would be sent to the model
                print(f"\n📝 Расширенный промпт для модели:")
                print("-" * 40)
                print(rag_result["enhanced_prompt"])
                print("-" * 40)
                
                # Simulate response
                concepts_count = len(rag_result.get("concepts_used", []))
                print(f"\n✅ Использована информация из базы знаний ({concepts_count} концепций)")
                
            else:
                print("\n💡 Релевантная информация не найдена. Генерирую ответ на основе модели...")
                print(f"\n📝 Оригинальный промпт: {prompt}")
        
        except Exception as e:
            print(f"\n❌ Ошибка при обработке: {e}")
        
        print(f"\n⏱ Время обработки: {rag_result.get('processing_time', 0):.2f}с")

async def test_neo4j_connection():
    """Test Neo4j connection"""
    print("🔗 Тестирование подключения к Neo4j...")
    
    try:
        neo4j = Neo4jConnection()
        result = neo4j.execute_query("RETURN 1 as test")
        if result:
            print("✅ Neo4j connection successful")
            return True
        else:
            print("❌ Neo4j connection failed - no result")
            return False
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Запуск тестов интеграции RAG в Telegram бот\n")
    
    # Test Neo4j connection first
    if not await test_neo4j_connection():
        print("\n❌ Не удалось подключиться к Neo4j. Проверьте настройки подключения.")
        return
    
    # Test RAG integration
    await test_bot_rag_integration()
    
    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(main()) 