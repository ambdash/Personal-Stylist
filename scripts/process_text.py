import spacy
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from neo4j import GraphDatabase
import logging
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Russian language model
nlp = spacy.load('ru_core_news_lg')

# Key node types to prioritize
KEY_NODE_TYPES = {'Случай', 'Эстетика', 'Сезон', 'Тренд', 'Погода'}

class Neo4jTextProcessor:
    def __init__(self):
        # Neo4j connection settings
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "password123"
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

    def get_text_spans(self, doc) -> set:
        """Extract meaningful text spans from document with Russian language support"""
        spans = set()
        
        # Add individual tokens and their lemmas
        for token in doc:
            if not token.is_punct and not token.is_space:
                spans.add(token.text.lower())
                spans.add(token.lemma_.lower())
                if 'ё' in token.text.lower():
                    spans.add(token.text.lower().replace('ё', 'е'))
                if 'ё' in token.lemma_.lower():
                    spans.add(token.lemma_.lower().replace('ё', 'е'))

        # Add meaningful phrases
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        for i in range(len(tokens)-1):
            if self._is_meaningful_phrase(tokens[i:i+2]):
                spans.add(' '.join([t.text.lower() for t in tokens[i:i+2]]))
                spans.add(' '.join([t.lemma_.lower() for t in tokens[i:i+2]]))

        for i in range(len(tokens)-2):
            if self._is_meaningful_phrase(tokens[i:i+3]):
                spans.add(' '.join([t.text.lower() for t in tokens[i:i+3]]))
                spans.add(' '.join([t.lemma_.lower() for t in tokens[i:i+3]]))

        return spans

    def _is_meaningful_phrase(self, tokens: List) -> bool:
        """Check if a sequence of tokens forms a meaningful phrase"""
        if not tokens:
            return False
        has_noun_or_adj = any(t.pos_ in {'NOUN', 'ADJ', 'PROPN'} for t in tokens)
        is_connected = all(t1.has_vector and t2.has_vector and 
                         t1.similarity(t2) > 0.3 
                         for t1, t2 in zip(tokens, tokens[1:]))
        return has_noun_or_adj and is_connected

    def _get_node_aliases(self):
        """Get common aliases for different node types"""
        return {
            'Случай': {
                'покупки': ['шопинг', 'поход по магазинам', 'шоппинг'],
                'прогулка': ['прогулки', 'гулять', 'променад'],
                'офис': ['работа', 'бизнес-встреча', 'деловая встреча'],
                'вечеринка': ['пати', 'party', 'праздник']
            },
            'Погода': {
                'переменчивая': ['изменчивая погода', 'переменчивая погода'],
                'солнечная': ['солнечно', 'солнечный день'],
                'дождливая': ['дождь', 'дождливый день']
            },
            'Эстетика': {
                '90-е': ['1990-е', 'девяностые', 'стиль 90-х'],
                'минимализм': ['минималистичный', 'минималистический'],
                'dark academia': ['дарк академия', 'темная академия'],
                'деловой стиль': ['офисный стиль', 'бизнес стиль']
            }
        }

    def find_key_nodes(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find key nodes with better alias handling and fuzzy matching"""
        aliases = self._get_node_aliases()
        
        with self.driver.session() as session:
            # First try exact matches including known aliases
            exact_query = """
            UNWIND $spans as span
            MATCH (n)
            WHERE n.name = span OR 
                  any(alias IN n.aliases WHERE alias = span)
            WITH n, labels(n) as nodeLabels
            WHERE any(label IN nodeLabels WHERE label IN $keyTypes)
            RETURN n.name as name,
                   n.id as id,
                   nodeLabels as labels,
                   2.0 as score
            """
            
            # Then try contains for remaining unmatched types
            contains_query = """
            UNWIND $spans as span
            MATCH (n)
            WHERE (n.name CONTAINS span OR 
                  any(alias IN n.aliases WHERE alias CONTAINS span))
            WITH n, labels(n) as nodeLabels, span
            WHERE any(label IN nodeLabels WHERE label IN $keyTypes)
            RETURN n.name as name,
                   n.id as id,
                   nodeLabels as labels,
                   1.0 as score,
                   span as matched_text
            """
            
            doc = nlp(text.lower())
            spans = set(self.get_text_spans(doc))
            
            # Add alias variations to spans
            for type_aliases in aliases.values():
                for main_term, alias_list in type_aliases.items():
                    if any(alias.lower() in text.lower() for alias in alias_list):
                        spans.add(main_term)
            
            # Get matches
            exact_results = session.run(exact_query, spans=list(spans), 
                                     keyTypes=list(KEY_NODE_TYPES)).data()
            contains_results = session.run(contains_query, spans=list(spans), 
                                        keyTypes=list(KEY_NODE_TYPES)).data()
            
            # Organize by node type, preferring exact matches
            organized = {type_: [] for type_ in KEY_NODE_TYPES}
            
            # Track what we've matched to avoid duplicates
            matched_nodes = set()
            
            # First add exact matches
            for node in exact_results:
                node_key = (node['name'], tuple(node['labels']))
                if node_key not in matched_nodes:
                    for label in node['labels']:
                        if label in KEY_NODE_TYPES:
                            organized[label].append(node)
                            matched_nodes.add(node_key)
            
            # Then add contains matches for types that don't have exact matches
            for node in contains_results:
                node_key = (node['name'], tuple(node['labels']))
                if node_key not in matched_nodes:
                    for label in node['labels']:
                        if label in KEY_NODE_TYPES and not organized[label]:
                            organized[label].append(node)
                            matched_nodes.add(node_key)
            
            return organized

    def _extract_item_type(self, text: str) -> Tuple[str, str]:
        """Extract specific item type and subtype from text if mentioned"""
        item_types = {
            'обувь': ['обувь', 'туфли', 'кроссовки', 'ботинки', 'сапоги', 'лоферы', 'балетки', 'оксфорды', 'босоножки'],
            'одежда': ['одежда', 'платье', 'юбка', 'брюки', 'джинсы', 'куртка', 'пальто', 'блуза', 'рубашка'],
            'аксессуары': ['аксессуары', 'сумка', 'украшения', 'серьги', 'браслет', 'колье', 'шарф', 'ремень']
        }
        
        doc = nlp(text.lower())
        
        # First try to find specific item mentions (e.g., "обувь:лоферы")
        for token in doc:
            for type_, keywords in item_types.items():
                for keyword in keywords:
                    if f"{type_}:{keyword}" in text.lower():
                        return type_, keyword
        
        # Then try to find general item type and specific subtype
        found_type = None
        found_subtype = None
        
        for word in doc:
            if not found_type:
                for type_, keywords in item_types.items():
                    if word.lemma_ == type_:
                        found_type = type_
                        break
            
            if not found_subtype:
                for type_, keywords in item_types.items():
                    if word.lemma_ in keywords and word.lemma_ != type_:
                        found_subtype = word.lemma_
                        if not found_type:
                            found_type = type_
                        break
        
        return found_type or "", found_subtype or ""

    def find_direct_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Find concepts that are directly mentioned in the text"""
        with self.driver.session() as session:
            doc = nlp(text)
            spans = self.get_text_spans(doc)
            
            # Extract item type and subtype
            item_type, item_subtype = self._extract_item_type(text)
            
            query_parts = ["MATCH (c:Концепт)"]
            where_conditions = []
            params = {"spans": list(spans)}
            
            # Add item type condition if found
            if item_type:
                if item_subtype:
                    where_conditions.append(f"(c.name STARTS WITH '{item_type}:{item_subtype}' OR c.name CONTAINS '{item_subtype}')")
                else:
                    where_conditions.append(f"c.name STARTS WITH '{item_type}:'")
            
            # Add span matching conditions
            where_conditions.append("""
                any(span IN $spans WHERE 
                    c.name CONTAINS span OR 
                    any(alias IN c.aliases WHERE alias CONTAINS span)
                )
            """)
            
            if where_conditions:
                query_parts.append("WHERE " + " OR ".join(where_conditions))
            
            query_parts.extend([
                "WITH DISTINCT c",
                "OPTIONAL MATCH (c)-[r:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(n)",
                "WITH c, collect({type: type(r), node: n}) as relations",
                """
                RETURN c.name as name,
                       c.id as id,
                       relations,
                       2.0 as score
                ORDER BY score DESC, c.name
                LIMIT 10
                """
            ])
            
            query = "\n".join(query_parts)
            return session.run(query, params).data()

    def format_concept_relations(self, concept: Dict[str, Any]) -> str:
        """Format concept relations in the desired output format"""
        relations = concept.get('relations', [])
        
        # Extract concept name
        name = concept['name']
        if ':' in name:
            name = name.split(':', 1)[1]  # Remove type prefix if exists
        
        # Group relations by type and node
        relation_parts = []
        
        # Map relation types to desired format
        relation_mapping = {
            'В_СЕЗОНЕ': 'в_сезоне',
            'ПОДХОДИТ_ДЛЯ': 'подходит_для',
            'ОТНОСИТСЯ_К': 'относится_к'
        }
        
        # Group relations by type
        relation_groups = {}
        for rel in relations:
            rel_type = rel['type']
            node = rel['node']
            if not node or 'name' not in node:
                continue
            
            mapped_type = relation_mapping.get(rel_type, rel_type.lower())
            if mapped_type not in relation_groups:
                relation_groups[mapped_type] = []
            relation_groups[mapped_type].append(node['name'])
        
        # Build the formatted string
        parts = [name]
        
        # Add relations in specific order
        for rel_type in ['в_сезоне', 'подходит_для', 'относится_к']:
            if rel_type in relation_groups:
                parts.append(f"{rel_type} {', '.join(relation_groups[rel_type])}")
        
        return ", ".join(parts)

    def process_text(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """Process text and find relevant concepts and relations"""
        start_time = time.time()
        
        try:
            # Find direct concept mentions with improved item type handling
            direct_concepts = self.find_direct_concepts(text)
            logger.info(f"Found {len(direct_concepts)} direct concept mentions")
            
            # Find and organize key nodes by type
            key_nodes = self.find_key_nodes(text)
            for type_, nodes in key_nodes.items():
                if nodes:
                    logger.info(f"Found {len(nodes)} {type_} nodes: {', '.join(n['name'] for n in nodes)}")
            
            # Format results
            formatted_results = []
            for i, concept in enumerate(direct_concepts, 1):
                formatted_result = f"{i}. {self.format_concept_relations(concept)}"
                formatted_results.append(formatted_result)
            
            # Add formatted results to the concept objects
            for concept, formatted in zip(direct_concepts, formatted_results):
                concept['formatted_output'] = formatted
            
            # Prepare output
            output = ["Вспомогательная информация:"] + formatted_results
            logger.info("\nFormatted output:")
            for line in output:
                logger.info(line)
            
            processing_time = time.time() - start_time
            return direct_concepts, output, processing_time
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

    def _has_relation_to_node(self, concept_id: str, node_id: str) -> bool:
        """Check if a concept has a relation to a specific node"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Концепт {id: $concept_id})-[:ОТНОСИТСЯ_К]->(n {id: $node_id})
                RETURN count(*) > 0 as has_relation
                """, concept_id=concept_id, node_id=node_id).single()
            return result and result['has_relation']

def save_results_to_csv(results: List[Dict], output_file: str):
    """Save processing results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    processor = Neo4jTextProcessor()
    try:
        test_file = Path('src/data/data/splits/test.jsonl')
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Create results files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_file = results_dir / f'processing_results_{timestamp}.jsonl'
        csv_file = results_dir / f'processing_summary_{timestamp}.csv'
        
        results_summary = []
        
        with open(test_file, 'r', encoding='utf-8') as f, \
             open(jsonl_file, 'w', encoding='utf-8') as out:
            
            for line in f:
                data = json.loads(line)
                prompt = data['instruction']
                
                logger.info(f"\nProcessing prompt: {prompt}")
                
                concepts, relations, processing_time = processor.process_text(prompt)
                
                # Log results
                logger.info(f"Found concepts: {len(concepts)}")
                for concept in concepts:
                    logger.info(f"- {concept['name']} (score: {concept['score']:.2f})")
                
                logger.info(f"\nFound relations: {len(relations)}")
                if relations:
                    logger.info("Relations found:")
                    for rel in relations:
                        if rel['type'] == 'combination':
                            logger.info(f"→ {rel['item1']} {rel['relation']} {rel['item2']} ({rel['type']})")
                        else:
                            logger.info(f"→ {rel['item1']} {rel['relation']} {rel['item2']} ({rel['type']})")
                
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                
                # Save detailed results
                result = {
                    'prompt': prompt,
                    'concepts': concepts,
                    'relations': relations,
                    'processing_time': processing_time
                }
                json.dump(result, out, ensure_ascii=False)
                out.write('\n')
                
                # Add to summary
                results_summary.append({
                    'prompt': prompt,
                    'n_concepts': len(concepts),
                    'n_relations': len(relations),
                    'n_bridge_relations': len([r for r in relations if r['type'] == 'combination']),
                    'processing_time': processing_time,
                    'key_concepts': ', '.join(c['name'] for c in concepts[:3]),
                    'has_key_nodes': any(any(label in KEY_NODE_TYPES for label in c.get('labels', [])) 
                                       for c in concepts)
                })
                
                logger.info("-" * 80)
        
        # Save summary to CSV
        save_results_to_csv(results_summary, csv_file)
        logger.info(f"\nResults saved to:\n- {jsonl_file}\n- {csv_file}")
    
    finally:
        processor.close()

if __name__ == "__main__":
    main() 