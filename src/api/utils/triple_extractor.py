import re
from typing import List, Tuple

class TripleExtractor:
    STYLES = ["Y2K", "grunge", "old money", "clean girl", "dark academia", "cottagecore"]
    EVENTS = ["вечеринка", "офис", "свидание", "фестиваль", "прогулка", "поездка", "путешествие", "встреча"]
    ITEMS = ["юбка", "топ", "джинсы", "свитер", "платье", "куртка", "рубашка", "ботинки", "обувь"]
    SEASONS = ["лето", "зима", "весна", "осень"]
    COLORS = ["розовый", "бежевый", "черный", "белый", "красный", "синий", "зеленый"]

    @staticmethod
    def extract_triples(text: str) -> List[Tuple[str, str, str]]:
        triples = []
        
        # Extract style-related triples
        for style in TripleExtractor.STYLES:
            if style.lower() in text.lower():
                triples.append((style, "ВКЛЮЧАЕТ", text))
                
        # Extract event-related triples
        for event in TripleExtractor.EVENTS:
            if event in text.lower():
                triples.append((text, "ПОДХОДИТ_ДЛЯ", event))
                
        # Extract item-related triples
        for item in TripleExtractor.ITEMS:
            if item in text.lower():
                triples.append((text, "СОДЕРЖИТ", item))
                
        # Extract season-related triples
        for season in TripleExtractor.SEASONS:
            if season in text.lower():
                triples.append((text, "ПРЕДПОЧТИТЕЛЕН_В", season))
                
        # Extract color-related triples
        for color in TripleExtractor.COLORS:
            if color in text.lower():
                triples.append((text, "СОДЕРЖИТ_ЦВЕТ", color))
                
        return triples 