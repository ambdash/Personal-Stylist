#!/usr/bin/env python3

import spacy
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Russian language model
nlp = spacy.load('ru_core_news_lg')

# Key node types to prioritize
KEY_NODE_TYPES = {'Случай', 'Эстетика', 'Сезон', 'Тренд', 'Погода'}

def get_text_spans(doc) -> set:
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
        spans.add(' '.join([t.text.lower() for t in tokens[i:i+2]]))
        spans.add(' '.join([t.lemma_.lower() for t in tokens[i:i+2]]))

    for i in range(len(tokens)-2):
        spans.add(' '.join([t.text.lower() for t in tokens[i:i+3]]))
        spans.add(' '.join([t.lemma_.lower() for t in tokens[i:i+3]]))

    return spans

def debug_enhanced_processor():
    text = "Какую летнюю обувь лучше выбрать для офиса в 2025 году?"
    
    print(f"Original text: {text}")
    print("="*60)
    
    # Process with spaCy
    doc = nlp(text.lower())
    spans = set(get_text_spans(doc))
    
    print("Extracted spans:")
    for span in sorted(spans):
        print(f"  - '{span}'")
    
    print("\n" + "="*60)
    print("DEBUGGING KEY NODE FINDING:")
    print("="*60)
    
    # Connect to database
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
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
                   2.0 as score,
                   span as matched_span
            """
            
            # Contains match query
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
                   span as matched_span
            """
            
            print("EXACT MATCHES:")
            exact_results = session.run(exact_query, spans=list(spans), 
                                     keyTypes=list(KEY_NODE_TYPES)).data()
            for result in exact_results:
                print(f"  {result['name']} ({result['labels']}) - matched span: '{result['matched_span']}'")
            
            print("\nCONTAINS MATCHES:")
            contains_results = session.run(contains_query, spans=list(spans), 
                                        keyTypes=list(KEY_NODE_TYPES)).data()
            for result in contains_results:
                print(f"  {result['name']} ({result['labels']}) - matched span: '{result['matched_span']}'")
            
            print("\n" + "="*60)
            print("CHECKING SPECIFIC EXPECTED NODES:")
            print("="*60)
            
            # Check specific nodes we expect
            expected_checks = [
                ("лето", "Сезон"),
                ("офис", "Случай"), 
                ("2025", "Тренд")
            ]
            
            for node_name, node_type in expected_checks:
                # Check if node exists
                result = session.run(f'''
                    MATCH (n:{node_type})
                    WHERE n.name = $name
                    RETURN n.name as name, n.id as id, n.aliases as aliases
                ''', name=node_name)
                
                nodes = result.data()
                if nodes:
                    node = nodes[0]
                    print(f"\n{node_type}:{node_name} exists:")
                    print(f"  ID: {node['id']}")
                    print(f"  Aliases: {node.get('aliases', [])}")
                    
                    # Check which spans would match this node
                    matching_spans = []
                    for span in spans:
                        if span == node_name or node_name in span or span in node_name:
                            matching_spans.append(span)
                        if node.get('aliases'):
                            for alias in node['aliases']:
                                if span == alias or alias in span or span in alias:
                                    matching_spans.append(span)
                    
                    if matching_spans:
                        print(f"  Matching spans: {matching_spans}")
                    else:
                        print(f"  No matching spans found")
                else:
                    print(f"\n{node_type}:{node_name} does not exist")
                    
    finally:
        driver.close()

if __name__ == "__main__":
    debug_enhanced_processor() 