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

# Lazy load Russian language model
nlp = None

def get_nlp():
    """Lazy load spaCy model"""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load('ru_core_news_lg')
            logger.info("Successfully loaded Russian spaCy model")
        except OSError as e:
            logger.error(f"Failed to load Russian spaCy model: {e}")
            # Fallback to a simpler model or disable NLP features
            try:
                nlp = spacy.load('ru_core_news_sm')
                logger.info("Loaded smaller Russian spaCy model as fallback")
            except OSError:
                logger.warning("No Russian spaCy model available, using basic text processing")
                nlp = None
    return nlp

# Key node types to prioritize
KEY_NODE_TYPES = {'Случай', 'Эстетика', 'Сезон', 'Тренд', 'Погода'}

class EnhancedTextProcessor:
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
            if not token.is_punct and not token.is_space and len(token.text) > 1:  # Filter out single letters
                spans.add(token.text.lower())
                spans.add(token.lemma_.lower())
                if 'ё' in token.text.lower():
                    spans.add(token.text.lower().replace('ё', 'е'))
                if 'ё' in token.lemma_.lower():
                    spans.add(token.lemma_.lower().replace('ё', 'е'))

        # Add meaningful phrases
        tokens = [t for t in doc if not t.is_punct and not t.is_space and len(t.text) > 1]
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

    def _extract_item_type(self, text: str) -> Tuple[str, str]:
        """Extract specific item type and subtype from text if mentioned"""
        item_types = {
            'обувь': ['обувь', 'туфли', 'кроссовки', 'ботинки', 'сапоги', 'лоферы', 'балетки', 'оксфорды', 'босоножки'],
            'одежда': ['одежда', 'платье', 'юбка', 'брюки', 'джинсы', 'куртка', 'пальто', 'блуза', 'рубашка', 'свитер', 'джемпер', 'пиджак', 'жакет', 'наряд'],
            'аксессуары': ['аксессуары', 'сумка', 'украшения', 'серьги', 'браслет', 'колье', 'шарф', 'ремень', 'клатч']
        }
        
        nlp_model = get_nlp()
        if nlp_model is None:
            # Fallback to simple text matching
            text_lower = text.lower()
            for type_, keywords in item_types.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        return type_, keyword
            return "", ""
        
        doc = nlp_model(text.lower())
        
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
                    if word.lemma_ in keywords:
                        if word.lemma_ == type_:
                            found_type = type_
                        else:
                            found_subtype = word.lemma_
                            found_type = type_
                        break
        
        return found_type or "", found_subtype or ""

    def _extract_styling_context(self, text: str) -> Tuple[bool, str]:
        """Extract styling context like 'стилизовать свитер' or 'носить пиджак'"""
        nlp_model = get_nlp()
        if nlp_model is None:
            # Fallback to simple text matching
            styling_verbs = ['стилизовать', 'носить', 'сочетать', 'комбинировать']
            text_lower = text.lower()
            for verb in styling_verbs:
                if verb in text_lower:
                    return True, ""
            return False, ""
        
        doc = nlp_model(text.lower())
        
        styling_verbs = ['стилизовать', 'носить', 'сочетать', 'комбинировать']
        
        for i, token in enumerate(doc):
            if token.lemma_ in styling_verbs:
                # Look for the next meaningful noun
                for j in range(i+1, min(i+4, len(doc))):  # Look ahead up to 3 words
                    next_token = doc[j]
                    if next_token.pos_ in ['NOUN', 'PROPN'] and not next_token.is_punct:
                        return True, next_token.lemma_
        
        return False, ""

    def _is_formal_context(self, text: str, key_nodes: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Check if the context is formal (theater, office, restaurant, etc.)"""
        formal_keywords = ['театр', 'офис', 'ресторан', 'деловой', 'работа', 'встреча', 'официальный']
        
        # Check direct text mentions
        text_lower = text.lower()
        for keyword in formal_keywords:
            if keyword in text_lower:
                return True
        
        # Check key nodes for formal contexts
        for node_type, nodes in key_nodes.items():
            for node in nodes:
                if node['name'] in formal_keywords:
                    return True
        
        return False

    def _filter_irrelevant_styles(self, concepts: List[Dict[str, Any]], is_formal: bool) -> List[Dict[str, Any]]:
        """Filter out irrelevant style concepts based on context"""
        if not is_formal:
            return concepts
        
        # Styles that are inappropriate for formal contexts
        informal_styles = ['grunge', 'streetwear', 'уличный стиль', '80-е', '90-е', 'punk', 'gothic']
        
        filtered_concepts = []
        for concept in concepts:
            relations = concept.get('relations', [])
            has_informal_style = False
            
            for rel in relations:
                node = rel.get('node', {})
                if node and 'name' in node:
                    node_name = node['name'].lower()
                    if any(style in node_name for style in informal_styles):
                        has_informal_style = True
                        break
            
            if not has_informal_style:
                filtered_concepts.append(concept)
        
        return filtered_concepts

    def find_key_nodes(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Find key nodes with better alias handling and fuzzy matching"""
        with self.driver.session() as session:
            nlp_model = get_nlp()
            if nlp_model is None:
                # Fallback to simple text processing
                spans = set(text.lower().split())
            else:
                doc = nlp_model(text.lower())
                spans = set(self.get_text_spans(doc))
            
            # Add semantic mappings for seasons
            season_mappings = {
                'летний': 'лето',
                'летнюю': 'лето', 
                'летняя': 'лето',
                'летнее': 'лето',
                'зимний': 'зима',
                'зимнюю': 'зима',
                'зимняя': 'зима',
                'зимнее': 'зима',
                'весенний': 'весна',
                'весеннюю': 'весна',
                'весенняя': 'весна',
                'весеннее': 'весна',
                'осенний': 'осень',
                'осеннюю': 'осень',
                'осенняя': 'осень',
                'осеннее': 'осень'
            }
            
            # Add season mappings to spans
            for span in list(spans):
                if span in season_mappings:
                    spans.add(season_mappings[span])
            
            # Filter out very short spans that cause noise
            spans = {span for span in spans if len(span) > 1}
            
            # Exact match query
            exact_query = """
            UNWIND $spans as span
            MATCH (n)
            WHERE n.name = span OR 
                  any(alias IN n.aliases WHERE alias = span)
            WITH n, labels(n) as nodeLabels, span
            WHERE any(label IN nodeLabels WHERE label IN $keyTypes)
            RETURN n.name as name,
                   n.id as id,
                   nodeLabels as labels,
                   2.0 as score
            """
            
            # Contains match query - only for longer spans to avoid noise
            contains_query = """
            UNWIND $spans as span
            MATCH (n)
            WHERE size(span) > 3 AND (n.name CONTAINS span OR 
                  any(alias IN n.aliases WHERE alias CONTAINS span))
            WITH n, labels(n) as nodeLabels, span
            WHERE any(label IN nodeLabels WHERE label IN $keyTypes)
            RETURN n.name as name,
                   n.id as id,
                   nodeLabels as labels,
                   1.0 as score
            """
            
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

    def find_concepts_by_item_type_and_key_nodes(self, text: str, item_type: str, key_nodes: Dict[str, List[Dict[str, Any]]], styling_item: str = "") -> List[Dict[str, Any]]:
        """Find concepts based on item type and key node intersections"""
        with self.driver.session() as session:
            # Build query for concepts with item type filter and key node intersections
            query_parts = ["MATCH (c:Концепт)"]
            where_conditions = []
            params = {}
            
            # Add item type condition - this is MANDATORY if item type is detected
            if item_type:
                if styling_item:
                    # For styling contexts, look for the specific item being styled
                    where_conditions.append(f"(c.name CONTAINS '{styling_item}' OR c.name STARTS WITH 'Концепт:{styling_item}')")
                else:
                    # For general item type mentions, be VERY strict about the category
                    if item_type == 'обувь':
                        item_type_conditions = [
                            f"c.name CONTAINS '{item_type}'",
                            f"c.name STARTS WITH 'Концепт:{item_type}'",
                            f"EXISTS((c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ]->())"
                        ]
                    elif item_type == 'одежда':
                        item_type_conditions = [
                            f"c.name CONTAINS '{item_type}'",
                            f"c.name STARTS WITH 'Концепт:{item_type}'",
                            f"EXISTS((c)-[:ЯВЛЯЕТСЯ_ОДЕЖДОЙ]->())"
                        ]
                    elif item_type == 'аксессуары':
                        item_type_conditions = [
                            f"c.name CONTAINS '{item_type}'",
                            f"c.name STARTS WITH 'Концепт:{item_type}'",
                            f"EXISTS((c)-[:ЯВЛЯЕТСЯ_АКСЕССУАРОМ]->())"
                        ]
                    else:
                        # Fallback for other item types
                        item_type_conditions = [
                            f"c.name CONTAINS '{item_type}'",
                            f"c.name STARTS WITH 'Концепт:{item_type}'"
                        ]
                    
                    where_conditions.append("(" + " OR ".join(item_type_conditions) + ")")
                    
                    # Add explicit exclusion of other item types when specific type is mentioned
                    if item_type == 'обувь':
                        where_conditions.append("NOT EXISTS((c)-[:ЯВЛЯЕТСЯ_ОДЕЖДОЙ|ЯВЛЯЕТСЯ_АКСЕССУАРОМ]->())")
                    elif item_type == 'одежда':
                        where_conditions.append("NOT EXISTS((c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ|ЯВЛЯЕТСЯ_АКСЕССУАРОМ]->())")
                    elif item_type == 'аксессуары':
                        where_conditions.append("NOT EXISTS((c)-[:ЯВЛЯЕТСЯ_ОБУВЬЮ|ЯВЛЯЕТСЯ_ОДЕЖДОЙ]->())")
            
            # Add key node intersection conditions - this is the main filter
            key_node_conditions = []
            for node_type, nodes in key_nodes.items():
                if nodes:
                    node_ids = [node['id'] for node in nodes]
                    params[f"{node_type.lower()}_ids"] = node_ids
                    key_node_conditions.append(f"""
                        EXISTS {{
                            MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                            WHERE target_node.id IN ${node_type.lower()}_ids
                        }}
                    """)
            
            if key_node_conditions:
                where_conditions.append("(" + " OR ".join(key_node_conditions) + ")")
            
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))
            
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
                LIMIT 2000
                """
            ])
            
            query = "\n".join(query_parts)
            return session.run(query, params).data()

    def find_direct_concepts(self, text: str, key_nodes: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find concepts that are directly mentioned in the text"""
        with self.driver.session() as session:
            nlp_model = get_nlp()
            if nlp_model is None:
                # Fallback to simple text processing
                spans = set(text.lower().split())
            else:
                doc = nlp_model(text)
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
            
            # Add key node intersection for better relevance
            key_node_conditions = []
            for node_type, nodes in key_nodes.items():
                if nodes:
                    node_ids = [node['id'] for node in nodes]
                    params[f"{node_type.lower()}_ids"] = node_ids
                    key_node_conditions.append(f"""
                        EXISTS {{
                            MATCH (c)-[:ОТНОСИТСЯ_К|ПОДХОДИТ_ДЛЯ|В_СЕЗОНЕ]->(target_node)
                            WHERE target_node.id IN ${node_type.lower()}_ids
                        }}
                    """)
            
            if key_node_conditions:
                where_conditions.append("(" + " OR ".join(key_node_conditions) + ")")
            
            if where_conditions:
                query_parts.append("WHERE " + " AND ".join(where_conditions))
            
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
                LIMIT 20
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

    def calculate_intersection_score(self, concept: Dict[str, Any], key_nodes: Dict[str, List[Dict[str, Any]]]) -> int:
        """Calculate how many key nodes this concept intersects with"""
        score = 0
        relations = concept.get('relations', [])
        
        # Get all related node IDs from the concept
        concept_related_ids = set()
        for rel in relations:
            node = rel.get('node')
            if node and 'id' in node:
                concept_related_ids.add(node['id'])
        
        # Count intersections with key nodes
        for node_type, nodes in key_nodes.items():
            for node in nodes:
                if node['id'] in concept_related_ids:
                    score += 1
        
        return score

    def _get_minimum_intersection_threshold(self, total_keynodes: int) -> int:
        """Get minimum intersection score based on total keynodes found"""
        if total_keynodes >= 3:
            return 2  # Need at least 2 keynode matches when 3+ keynodes are found
        elif total_keynodes == 2:
            return 1  # Need at least 1 keynode match when 2 keynodes are found
        else:
            return 0  # No minimum when few keynodes are found

    def process_text(self, text: str) -> Tuple[str, List[Dict[str, Any]], float]:
        """Process text and return formatted prompt with auxiliary information"""
        start_time = time.time()
        
        try:
            # Extract item type
            item_type, item_subtype = self._extract_item_type(text)
            logger.info(f"Detected item type: {item_type}, subtype: {item_subtype}")
            
            # Extract styling context
            is_styling, styling_item = self._extract_styling_context(text)
            if is_styling:
                logger.info(f"Detected styling context for: {styling_item}")
            
            # Find key nodes
            key_nodes = self.find_key_nodes(text)
            total_keynodes = sum(len(nodes) for nodes in key_nodes.values())
            
            for type_, nodes in key_nodes.items():
                if nodes:
                    logger.info(f"Found {len(nodes)} {type_} nodes: {', '.join(n['name'] for n in nodes)}")
            
            # Find concepts based on strategy
            if item_type or is_styling:
                # Strategy 1: Direct mention of clothing/shoes/accessories or styling context
                logger.info("Using strategy 1: Direct item type mention or styling context")
                styling_target = styling_item if is_styling else ""
                concepts = self.find_concepts_by_item_type_and_key_nodes(text, item_type, key_nodes, styling_target)
            else:
                # Strategy 2: Only key node intersections
                logger.info("Using strategy 2: Key node intersections only")
                concepts = self.find_direct_concepts(text, key_nodes)
            
            # Calculate intersection scores and apply filtering
            for concept in concepts:
                concept['intersection_score'] = self.calculate_intersection_score(concept, key_nodes)
            
            # Apply minimum intersection threshold
            min_threshold = self._get_minimum_intersection_threshold(total_keynodes)
            if min_threshold > 0:
                concepts = [c for c in concepts if c['intersection_score'] >= min_threshold]
                logger.info(f"Applied minimum intersection threshold: {min_threshold}, remaining concepts: {len(concepts)}")
            
            # Sort by intersection score (descending), then by name
            concepts.sort(key=lambda x: (-x['intersection_score'], x['name']))
            
            logger.info(f"Found {len(concepts)} concepts after filtering")
            
            # Format results
            formatted_results = []
            for i, concept in enumerate(concepts[:5], 1):  # Top 5 concepts
                formatted_result = f"{i}. {self.format_concept_relations(concept)}"
                formatted_results.append(formatted_result)
                logger.info(f"Concept {i}: {concept['name']} (intersection score: {concept['intersection_score']})")
            
            # Build final prompt
            if formatted_results:
                auxiliary_info = ["Вспомогательная информация:"] + formatted_results
                final_prompt = text + "\n\n" + "\n".join(auxiliary_info)
            else:
                final_prompt = text
            
            processing_time = time.time() - start_time
            return final_prompt, concepts, processing_time
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            raise

def save_results_to_csv(results: List[Dict], output_file: str):
    """Save processing results to CSV file"""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    processor = EnhancedTextProcessor()
    try:
        test_file = Path('src/data/data/splits/test.jsonl')
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Create results files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        jsonl_file = results_dir / f'processing_results_{timestamp}.jsonl'
        csv_file = results_dir / f'processing_summary_{timestamp}.csv'
        processed_prompts_file = results_dir / f'processed_prompts_{timestamp}.jsonl'
        
        results_summary = []
        
        with open(test_file, 'r', encoding='utf-8') as f, \
             open(jsonl_file, 'w', encoding='utf-8') as out, \
             open(processed_prompts_file, 'w', encoding='utf-8') as prompts_out:
            
            for line in f:
                data = json.loads(line)
                prompt = data['instruction']
                
                logger.info(f"\nProcessing prompt: {prompt}")
                
                final_prompt, concepts, processing_time = processor.process_text(prompt)
                
                # Log results
                logger.info(f"Found concepts: {len(concepts)}")
                for concept in concepts[:3]:  # Show top 3
                    logger.info(f"- {concept['name']} (intersection score: {concept.get('intersection_score', 0)})")
                
                logger.info(f"Processing time: {processing_time:.2f} seconds")
                
                # Save detailed results
                result = {
                    'original_prompt': prompt,
                    'final_prompt': final_prompt,
                    'concepts': concepts,
                    'processing_time': processing_time
                }
                json.dump(result, out, ensure_ascii=False)
                out.write('\n')
                
                # Save processed prompt for inference
                processed_result = {
                    'instruction': final_prompt,
                    'output': data.get('output', ''),  # Keep original output if exists
                    'original_instruction': prompt
                }
                json.dump(processed_result, prompts_out, ensure_ascii=False)
                prompts_out.write('\n')
                
                # Add to summary
                results_summary.append({
                    'original_prompt': prompt,
                    'final_prompt': final_prompt,
                    'n_concepts': len(concepts),
                    'processing_time': processing_time,
                    'key_concepts': ', '.join(c['name'] for c in concepts[:3]),
                    'has_auxiliary_info': 'Вспомогательная информация:' in final_prompt
                })
                
                logger.info("-" * 80)
        
        # Save summary to CSV
        save_results_to_csv(results_summary, csv_file)
        logger.info(f"\nResults saved to:")
        logger.info(f"- Detailed results: {jsonl_file}")
        logger.info(f"- Summary: {csv_file}")
        logger.info(f"- Processed prompts for inference: {processed_prompts_file}")
    
    finally:
        processor.close()

if __name__ == "__main__":
    main() 