import argparse
import json
import csv
import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Set
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — ассистент по извлечению сущностей и связей для графовой базы знаний Neo4j. Твоя задача — из одного ответа модели (output) извлечь ключевые связи между базовыми категориями и их концептуальными деталями.

Типы узлов
Базовые категории (без признаков):
Одежда:* — примеры: Одежда:платье, Одежда:рубашка
Обувь:* — Обувь:ботинки
Аксессуар:* — Аксессуар:сумка

Концепты с признаками (не базовые):

Пример: Концепт:длинное хлопковое платье с цветочным принтом

⚠️ Не создавай концепты, полностью совпадающие с базовыми, например: Концепт:бомбер → ❌ (лучше сразу Одежда:бомбер)
Атрибуты и категории:

Материал:*, Цвет:*, Эстетика:*, Сезон:*, Погода:*, Случай:*, Тренд:*, Деталь:*

🔗 Допустимые отношения
Используй только указанные связи. Новые не придумывать!
ЯВЛЯЕТСЯ_ОДЕЖДОЙ → для связи с Одежда:*
ЯВЛЯЕТСЯ_ОБУВЬЮ → с Обувь:*
ЯВЛЯЕТСЯ_АКСЕССУАРОМ → с Аксессуар:*
СДЕЛАН_ИЗ → с Материал:*
ОКРАШЕН_В → с Цвет:*
ОТНОСИТСЯ_К → с Эстетика:*
ПОДХОДИТ_ДЛЯ → с Случай:* или Погода:*
В_СЕЗОНЕ → с Сезон:*
ИМЕЕТ_ПРИНТ → с Тренд:*
ИМЕЕТ_ДЕТАЛЬ → с Деталь:*
СОЧЕТАЕТСЯ_С → с другим Концепт:* (если явно указано сочетание)

✅ Список сезонов (нормализовать при совпадении с вариациями):
СЕЗОНЫ = [зима, весна, лето, осень]
✅ Список погодных условий (нормализовать при совпадении с синонимами):
ПОГОДА = [жаркая, солнечная, холодная, дождь, ветреная, морозная, переменчивая, слякотная, влажная]
✅ Список случаев (событий) (нормализовать при совпадении с синонимами):
СЛУЧАИ = [
  офис, встреча с друзьями, путешествие, вечеринка, ужин в ресторане,
  свадьба, романтическая встреча, пикник, прогулка, отпуск,
  работа, повседневная жизнь, фотосессия, шопинг, пляж
]
✅ Список эстетик (нормализовать при совпадении с alias):
ЭСТЕТИКИ = [
  quiet luxury, ретро-футуризм, casual, деловой стиль, уличный стиль,
  grunge, 70-е, 80-е, 90-е, y2k, dark academia, coquette,
  бохо, романтика, Pinterest, cottagecore, преппи, минимализм
]
🧩 Нормализация эстетик
Используй Эстетика:*, подставляя ключ из словаря AESTHETIC_ALIASES.
Пример:

"тихая роскошь" → Эстетика:quiet luxury

"прибрежная бабушка" → Эстетика:cottagecore

🌿 Нормализация сезонов
Используй только:

Сезон:зима, Сезон:весна, Сезон:лето, Сезон:осень

Все вариации вроде "знойное лето", "летний день", "весенняя погода" → нормализуй по словарю SEASONS.

🌦 Нормализация погоды
Только значения из WEATHER_SYNONYMS → Погода:*

Пример:

"жаркий день", "зной" → Погода:жаркая

🎯 Нормализация случая (события/повода)
Используй только значения из OCCASIONS. Все синонимы приводятся к ключу.
Пример:

"на работу", "офисный" → Случай:офис

"шопинг", "поход по магазинам" → Случай:шопинг

Пример:
Вопрос: Какой стильный аутфит подойдет для теплых осенних дней?

Output: Для теплых осенних дней отлично подойдет комплект из легкого трикотажного платья миди, кожаной куртки и ботинок на массивной подошве. Добавьте шарф нейтрального оттенка и солнцезащитные очки — так образ будет стильным и комфортным.

Триплеты:
Концепт:легкое трикотажное платье миди | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:платье  
Концепт:легкое трикотажное платье миди | СДЕЛАН_ИЗ | Материал:трикотаж  
Концепт:легкое трикотажное платье миди | В_СЕЗОНЕ | Сезон:осень  
Концепт:легкое трикотажное платье миди | ПОДХОДИТ_ДЛЯ | Погода:теплая  

Концепт:кожаная куртка | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:куртка  
Концепт:кожаная куртка | СДЕЛАН_ИЗ | Материал:кожа  
Концепт:кожаная куртка | В_СЕЗОНЕ | Сезон:осень  
Концепт:кожаная куртка | ПОДХОДИТ_ДЛЯ | Погода:теплая  

Концепт:ботинки на массивной подошве | ЯВЛЯЕТСЯ_ОБУВЬЮ | Обувь:ботинки  
Концепт:ботинки на массивной подошве | ИМЕЕТ_ДЕТАЛЬ | Деталь:массивная подошва  
Концепт:ботинки на массивной подошве | В_СЕЗОНЕ | Сезон:осень  

Концепт:шарф нейтрального оттенка | ЯВЛЯЕТСЯ_АКСЕССУАРОМ | Аксессуар:шарф  
Концепт:шарф нейтрального оттенка | ОКРАШЕН_В | Цвет:нейтральный  
Концепт:шарф нейтрального оттенка | В_СЕЗОНЕ | Сезон:осень  

Концепт:солнцезащитные очки | ЯВЛЯЕТСЯ_АКСЕССУАРОМ | Аксессуар:очки  
Концепт:солнцезащитные очки | ПОДХОДИТ_ДЛЯ | Погода:солнечная  

❌ Чего не делать:
Не создавать бессмысленных связей.
Не придумывать новые значения для Сезон, Погода, Случай, Эстетика.
Не дублировать базовую одежду в концепт (например: Концепт:бомбер | ЯВЛЯЕТСЯ_ОДЕЖДОЙ"""


#  SYSTEM_PROMPT = """Ты — ассистент по извлечению сущностей и связей для графовой базы знаний Neo4j. Твоя задача — из одного ответа модели (output) извлечь ключевые связи между базовыми категориями и их концептуальными деталями.

# ⚠️ ВАЖНО: В базе данных есть два уровня узлов:
# 1. Базовые категории - простые предметы без описаний:
#    - Одежда:платье
#    - Одежда:юбка
#    - Обувь:ботинки
#    - и т.д.

# 2. Концепты - конкретные реализации с описаниями:
#    - Концепт:длинное хлопковое платье
#    - Концепт:ботинки на шнуровке
#    - Концепт:вязаный кардиган в нейтральных тонах

# ❗ Правило: Если предмет имеет любые описания (цвет, материал, длина и т.д.) - он ДОЛЖЕН быть концептом!

# Пример правильного разбора:

# Input: Надень длинное хлопковое платье с цветочным принтом и вязаный кардиган в нейтральных тонах — идеальный образ в стиле cottagecore.

# Ответ:
# Концепт:длинное хлопковое платье с цветочным принтом | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:платье
# Концепт:длинное хлопковое платье с цветочным принтом | СДЕЛАН_ИЗ | Материал:хлопок
# Концепт:длинное хлопковое платье с цветочным принтом | ИМЕЕТ_ПРИНТ | Тренд:цветочный принт
# Концепт:длинное хлопковое платье с цветочным принтом | ОТНОСИТСЯ_К | Эстетика:cottagecore

# Концепт:вязаный кардиган в нейтральных тонах | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:кардиган
# Концепт:вязаный кардиган в нейтральных тонах | ОКРАШЕН_В | Цвет:нейтральный
# Концепт:вязаный кардиган в нейтральных тонах | ОТНОСИТСЯ_К | Эстетика:cottagecore

# Концепт:длинное хлопковое платье с цветочным принтом | СОЧЕТАЕТСЯ_С | Концепт:вязаный кардиган в нейтральных тонах

# ❌ Неправильно:
# Одежда:платье | ОТНОСИТСЯ_К | Эстетика:cottagecore
# (нельзя связывать базовые категории напрямую с характеристиками)

# ✅ Правильно:
# Концепт:длинное платье | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:платье
# Концепт:длинное платье | ОТНОСИТСЯ_К | Эстетика:cottagecore

# Обязательные префиксы для узлов:
# - Одежда: (только базовые предметы без описаний)
# - Обувь: (только базовые предметы без описаний)
# - Аксессуар: (только базовые предметы без описаний)
# - Концепт: (для всех предметов с описаниями)
# - Эстетика: (для стилей)
# - Цвет: (для цветов и оттенков)
# - Материал: (для тканей и материалов)
# - Случай: (для событий)
# - Погода: (для погодных условий)
# - Сезон: (для времен года)
# - Тренд: (для трендов и принтов)


# Пример правильного разбора:
# Ответ:
# Концепт:серебристый бархатный топ | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:топ
# Концепт:серебристый бархатный топ | ОКРАШЕН_В | Цвет:серебристый
# Концепт:серебристый бархатный топ | СДЕЛАН_ИЗ | Материал:бархат
# Концепт:серебристый бархатный топ | ПОДХОДИТ_ДЛЯ | Случай:вечеринка
# Концепт:серебристый бархатный топ | ОТНОСИТСЯ_К | Эстетика:y2k
# Концепт:джинсы с низкой посадкой | ЯВЛЯЕТСЯ_ОДЕЖДОЙ | Одежда:джинсы
# Концепт:серебристый бархатный топ | СОЧЕТАЕТСЯ_С | Концепт:джинсы с низкой посадкой

# Output: Надень серебристый топ из бархата и джинсы с низкой посадкой — идеальный образ для вечеринки в стиле y2k.

# Возможные отношения:
# - ЯВЛЯЕТСЯ_ОДЕЖДОЙ (от концепта к базовой одежде)
# - ЯВЛЯЕТСЯ_ОБУВЬЮ (от концепта к базовой обуви)
# - ЯВЛЯЕТСЯ_АКСЕССУАРОМ (от концепта к базовому аксессуару)
# - СДЕЛАН_ИЗ (к материалу)
# - ОКРАШЕН_В (к цвету)
# - ОТНОСИТСЯ_К (к эстетике)
# - ПОДХОДИТ_ДЛЯ (к случаю/погоде)
# - В_СЕЗОНЕ (к сезону)
# - ИМЕЕТ_ПРИНТ (к тренду/принту)
# - СОЧЕТАЕТСЯ_С (между концептами)

# ⚠️ Правила:
# 1. Если предмет имеет описания - создавай концепт
# 2. Базовые категории используй только для простых предметов
# 3. Все характеристики (цвет, материал и т.д.) связывай только с концептами
# 4. Используй существительные в именительном падеже
# 5. Избегай дублирования связей"""

class Neo4jRelationsGenerator:
    def __init__(self, input_file: str, output_dir: str, test_mode: bool = True):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url="https://api.proxyapi.ru/openai/v1"
        )
        
        # Create error logs directory
        self.error_dir = self.output_dir / "errors"
        self.error_dir.mkdir(exist_ok=True)
        
        # Initialize sets to track unique nodes and edges
        self.nodes = set()
        self.edges = set()
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing nodes and edges from CSV files if they exist"""
        nodes_file = self.output_dir / 'new_nodes.csv'
        edges_file = self.output_dir / 'new_edges.csv'
        
        if nodes_file.exists():
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.nodes.add((row['id'], row['name'], row['label']))
        
        if edges_file.exists():
            with open(edges_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.edges.add((row['start'], row['end'], row['relation']))

    def load_qa_data(self) -> List[Dict]:
        """Load QA pairs from JSON file"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)["filtered_data"]
            if self.test_mode:
                # Take only first 12 items in test mode
                return data[:12]
            return data

    async def extract_relations(self, instruction: str, output: str, index: int) -> List[str]:
        """Extract relations using OpenAI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Instruction: {instruction}\nOutput: {output}"}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            relations = response.choices[0].message.content.strip().split('\n')
            
            # Try to process relations immediately
            try:
                self.process_relations(relations)
                return relations
            except Exception as e:
                # If processing fails, save to error log
                error_file = self.error_dir / f"error_{index}.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(f"Instruction: {instruction}\nOutput: {output}\n\nRelations:\n")
                    f.write('\n'.join(relations))
                    f.write(f"\n\nError: {str(e)}")
                logger.error(f"Failed to process relations for index {index}: {e}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract relations for index {index}: {e}")
            return []

    def normalize_node_name(self, name: str) -> str:
        """Normalize node names to be in nominative case"""
        # Add basic word form normalization rules
        replacements = {
            'бомбером': 'бомбер',
            'топом': 'топ',
            'джинсами': 'джинсы',
            'ботинками': 'ботинки',
            'кардиганом': 'кардиган',
            'платьем': 'платье',
            'юбкой': 'юбка',
            'рубашкой': 'рубашка',
            # Add more replacements as needed
        }
        
        for old, new in replacements.items():
            if name.endswith(old):
                name = name[:-len(old)] + new
        
        return name

    def process_relations(self, relations: List[str]):
        """Process relations and update nodes/edges sets"""
        for relation in relations:
            try:
                # Skip empty lines
                if not relation.strip():
                    continue
                    
                # Debug logging
                logger.debug(f"Processing relation: {relation}")
                
                node1, relation_type, node2 = self.parse_relation(relation)
                if not all([node1, relation_type, node2]):
                    logger.warning(f"Invalid relation format: {relation}")
                    continue
                
                # Normalize node names
                try:
                    # Split and validate node1
                    if ':' not in node1:
                        logger.warning(f"Invalid node1 format (missing prefix): {node1}")
                        continue
                    node1_prefix, node1_name = node1.split(':', 1)
                    node1_name = self.normalize_node_name(node1_name)
                    node1 = f"{node1_prefix}:{node1_name}"
                    
                    # Split and validate node2
                    if ':' not in node2:
                        logger.warning(f"Invalid node2 format (missing prefix): {node2}")
                        continue
                    node2_prefix, node2_name = node2.split(':', 1)
                    node2_name = self.normalize_node_name(node2_name)
                    node2 = f"{node2_prefix}:{node2_name}"
                    
                    # Validate that characteristics are only connected to concepts
                    if (node1_prefix in ['Цвет', 'Материал', 'Эстетика', 'Случай', 'Погода', 'Сезон', 'Тренд'] and 
                        node2_prefix not in ['Концепт']):
                        logger.warning(f"Invalid relation: characteristics can only connect to concepts")
                        continue
                        
                    if (node2_prefix in ['Цвет', 'Материал', 'Эстетика', 'Случай', 'Погода', 'Сезон', 'Тренд'] and 
                        node1_prefix not in ['Концепт']):
                        logger.warning(f"Invalid relation: characteristics can only connect to concepts")
                        continue
                    
                    # Add to sets
                    self.nodes.add((node1, node1_name, node1_prefix))
                    self.nodes.add((node2, node2_name, node2_prefix))
                    self.edges.add((node1, node2, relation_type))
                    
                    logger.debug(f"Successfully processed: {node1} -> {relation_type} -> {node2}")
                    
                except Exception as e:
                    logger.warning(f"Error processing node names: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing relation '{relation}': {str(e)}")
                continue

    def parse_relation(self, relation: str) -> Tuple[str, str, str]:
        """Parse a relation string into components"""
        try:
            node1, relation_type, node2 = [part.strip() for part in relation.split('|')]
            return node1, relation_type, node2
        except:
            return None, None, None

    def save_to_csv(self):
        """Save current nodes and edges to CSV files"""
        try:
            # Save nodes
            with open(self.output_dir / 'new_nodes.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'name', 'label'])
                for node_data in sorted(self.nodes):
                    writer.writerow(node_data)
            
            # Save edges
            with open(self.output_dir / 'new_edges.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['start', 'end', 'relation'])
                writer.writerows(sorted(self.edges))
            
            # Log statistics
            logger.info(f"Saved {len(self.nodes)} nodes and {len(self.edges)} edges")
            
            # Generate updated Cypher import script
            cypher_script = """// Импорт узлов
LOAD CSV WITH HEADERS FROM 'file:///new_nodes.csv' AS row
MERGE (n:`${row.label}` {id: row.id})
  ON CREATE SET n.name = row.name;

// Импорт связей
LOAD CSV WITH HEADERS FROM 'file:///new_edges.csv' AS row
MATCH (a {id: row.start})
MATCH (b {id: row.end})
MERGE (a)-[r:`${row.relation}`]->(b);
"""
            with open(self.output_dir / 'new_import.cypher', 'w', encoding='utf-8') as f:
                f.write(cypher_script)
                
        except Exception as e:
            logger.error(f"Error saving CSV files: {str(e)}")
            raise

    async def process_dataset(self):
        """Process the entire dataset"""
        data = self.load_qa_data()
        processed_count = 0
        
        # Process every 3rd pair
        for i in range(0, len(data), 2):
            if self.test_mode and processed_count >= 4:
                logger.info("Test mode: stopping after 4 examples")
                break
                
            pair = data[i]
            logger.info(f"Processing pair {i} (example {processed_count + 1})")
            
            try:
                relations = await self.extract_relations(
                    pair['instruction'],
                    pair['output'],
                    i
                )
                
                if relations:
                    processed_count += 1
                    # Save after each successful processing
                    self.save_to_csv()
                    
            except Exception as e:
                logger.error(f"Error processing pair {i}: {str(e)}")
                continue
        
        logger.info(f"Processed {processed_count} examples")
        logger.info(f"Generated Neo4j files in {self.output_dir}")
        if any(self.error_dir.iterdir()):
            logger.warning(f"Some relations failed to process. Check error logs in {self.error_dir}")

async def main():

    parser = argparse.ArgumentParser(description='Generate Neo4j relations from fashion QA data')

    parser.add_argument('--input', type=str,
        default=str(PROJECT_ROOT / 'src/data/data/filtered_fashion_qa.json'),
        help='Path to input QA JSON file')

    parser.add_argument('--output-dir', type=str,
        default=str(PROJECT_ROOT / 'src/data/neo4j_data'),
        help='Directory for output files')
    parser.add_argument('--test', action='store_true',
                      help='Run in test mode (process only 4 examples)')
    
    args = parser.parse_args()
    
    try:
        generator = Neo4jRelationsGenerator(args.input, args.output_dir, test_mode=args.test)
        await generator.process_dataset()
    except Exception as e:
        logger.error(f"Failed to generate Neo4j relations: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 