#!/usr/bin/env python3

import spacy
import pymorphy3
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model
nlp = spacy.load("ru_core_news_sm")
morph = pymorphy3.MorphAnalyzer()

def get_text_spans(doc) -> set:
    """Extract meaningful spans from text"""
    spans = set()
    
    # Single tokens
    for token in doc:
        if not token.is_stop and not token.is_punct and len(token.text) > 2:
            # Add original form
            spans.add(token.text.lower())
            
            # Add lemma
            spans.add(token.lemma_.lower())
            
            # Add pymorphy3 normal form
            parsed = morph.parse(token.text)[0]
            spans.add(parsed.normal_form.lower())
    
    # Add bigrams and trigrams manually since noun_chunks isn't available
    tokens = [token for token in doc if not token.is_stop and not token.is_punct and len(token.text) > 2]
    
    # Bigrams
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i].text.lower()} {tokens[i+1].text.lower()}"
        spans.add(bigram)
        
        # Lemmatized bigram
        lemma_bigram = f"{tokens[i].lemma_.lower()} {tokens[i+1].lemma_.lower()}"
        spans.add(lemma_bigram)
    
    # Trigrams
    for i in range(len(tokens) - 2):
        trigram = f"{tokens[i].text.lower()} {tokens[i+1].text.lower()} {tokens[i+2].text.lower()}"
        spans.add(trigram)
        
        # Lemmatized trigram
        lemma_trigram = f"{tokens[i].lemma_.lower()} {tokens[i+1].lemma_.lower()} {tokens[i+2].lemma_.lower()}"
        spans.add(lemma_trigram)
    
    # Named entities
    for ent in doc.ents:
        if len(ent.text.strip()) > 2:
            spans.add(ent.text.lower().strip())
    
    return spans

def debug_text_processing():
    text = "Какую летнюю обувь лучше выбрать для офиса в 2025 году?"
    
    print(f"Original text: {text}")
    print("="*60)
    
    # Process with spaCy
    doc = nlp(text)
    spans = get_text_spans(doc)
    
    print("Extracted spans:")
    for span in sorted(spans):
        print(f"  - '{span}'")
    
    print("\n" + "="*60)
    print("CHECKING DATABASE MATCHES:")
    print("="*60)
    
    # Connect to database
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    try:
        with driver.session() as session:
            # Check which concepts match our spans
            for span in sorted(spans):
                result = session.run('''
                    MATCH (c:Концепт) 
                    WHERE c.name CONTAINS $span
                    RETURN c.name as name
                    ORDER BY c.name
                    LIMIT 5
                ''', span=span)
                
                concepts = result.data()
                if concepts:
                    print(f"\nSpan '{span}' matches:")
                    for concept in concepts:
                        print(f"  - {concept['name']}")
            
            print("\n" + "="*60)
            print("CHECKING SPECIFIC SHOE CONCEPTS:")
            print("="*60)
            
            # Check if our target concepts would match any spans
            target_concepts = [
                'легкие кожаные лоферы',
                'бежевые балетки', 
                'строгие ботинки-оксфорды'
            ]
            
            for concept_name in target_concepts:
                print(f"\nConcept: {concept_name}")
                matches = []
                for span in spans:
                    if span in concept_name.lower() or concept_name.lower().find(span) != -1:
                        matches.append(span)
                
                if matches:
                    print(f"  Matching spans: {matches}")
                else:
                    print("  No matching spans found")
                    
                # Check what words are in the concept name
                concept_words = concept_name.lower().split()
                print(f"  Concept words: {concept_words}")
                print(f"  Words in spans: {[word for word in concept_words if word in spans]}")
                
    finally:
        driver.close()

if __name__ == "__main__":
    debug_text_processing() 