import json
import spacy
from spacy.pipeline import EntityRuler
from pathlib import Path
from collections import defaultdict
import re
import logging
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load spaCy model with only necessary components
nlp = spacy.load("ru_core_news_lg", disable=["parser", "ner"])

# === Define constants ===
SEASONS = {
    "зима": ["зимний", "зимняя", "зимнее", "зимние"],
    "весна": ["весенний", "весенняя", "весеннее", "весенние"],
    "лето": ["летний", "летняя", "летнее", "летние"],
    "осень": ["осенний", "осенняя", "осеннее", "осенние"]
}

OCCASIONS = {
    "офис": ["офис", "офисный", "деловая встреча", "работа", "рабочий"],
    "встреча с друзьями": ["встреча с друзьями", "дружеская встреча", "встреча друзей"],
    "путешествие": ["путешествие", "поездка", "отпуск", "поход"],
    "вечеринка": ["вечеринка", "party", "праздник", "торжество"],
    "ужин в ресторане": ["ужин в ресторане", "ресторан", "кафе", "ужин"],
    "свадьба": ["свадьба", "свадебный", "свадебная", "бракосочетание"],
    "романтическая встреча": ["романтическая встреча", "свидание", "романтический вечер", "романтика"],
    "пикник": ["пикник", "отдых на природе", "загородный отдых"],
    "прогулка": ["прогулка", "вечерняя прогулка", "дневная прогулка", "прогулка по городу", "городская прогулка"],
    "отпуск": ["отпуск", "отдых", "каникулы", "выходные"],
    "работа": ["работа", "офис", "деловая среда"],
    "повседневная жизнь": ["повседневная жизнь", "повседневный", "каждый день", "на каждый день"],
    "фотосессия": ["фотосессия", "фотосъемка", "съемка", "фотографирование"],
    "шопинг": ["шопинг", "покупки", "поход по магазинам", "shopping"],
    "пляж": ["пляж", "пляжный отдых", "морской отдых", "отдых у моря"]
}

WEATHER_SYNONYMS = {
    "жаркая": ["жаркий", "жара", "знойный", "тёплый летний", "зной"],
    "солнечная": ["солнечно", "яркое солнце", "ясная погода", "солнце", "солнечный день"],
    "холодная": ["холодно", "мороз", "прохладно", "низкая температура", "холодный"],
    "дождь": ["дождливо", "ливень", "моросящий дождь", "осадки", "дождливая"],
    "ветреная": ["ветер", "ветреная", "порывы ветра", "ветреный"],
    "морозная": ["мороз", "морозная погода", "морозный", "морозы"],
    "переменчивая": ["переменчивая погода", "нестабильная погода", "переменчивый"],
    "слякотная": ["слякоть", "грязь", "мокрый снег", "слякотный"],
    "влажная": ["влажность", "сырая погода", "влажный", "сырость"]
}

AESTHETIC_ALIASES =  {
    "quiet luxury": [
        "quiet luxury", "тихая роскошь", "тихий шик", "спокойная роскошь", 
        "спокойный роскошь", "тихая утонченность", "understated luxury", 
        "quietluxury", "quiet lux", "quiet luxuary"
    ],
    "ретро-футуризм": [
        "ретро-футуризм", "ретрофутуризм", "retro-futurism", "винтажный футуризм",
        "диско-футуризм", "disco futurism", "futurism"
    ],
    "casual": [
        "casual", "кэжуал", "универсальный стиль", "универсальный", 
        "непринужденный стиль", "повседневный стиль", "городская мода",
        "комфортный стиль", "повседневный", "casual chic", "casual-business",
        "smart casual", "relaxed casual", "умеренный кэжуал"
    ],
    "деловой стиль": [
        "деловой стиль", "деловой", "офисный стиль", "официальный стиль",
        "офисный", "business casual", "smart-casual", "офисная эстетика",
        "офисная", "деловой кэжуал"
    ],
    "уличный стиль": [
        "уличный стиль", "streetwear", "уличная мода", "уличная эстетика",
        "урбан", "urban", "street style", "уличный шик", "уличная культура",
        "street luxury"
    ],
    "grunge": [
        "grunge", "гранж", "гранжевая романтика", "грунд", "soft grunge",
        "мягкий гранж", "гранжевый", "гранж '90-х", "'90s grunge"
    ],
    "70-е": [
        "70-е", "диско 70-х", "диско 1970-х", "винтаж 70-х", "ретро 70-х",
        "'70s", "Эстетика:70-е диско"
    ],
    "80-е": [
        "80-е", "восьмидесятые", "мода 80-х", "'80s"
    ],
    "90-е": [
        "90-е", "девяностые", "мода 90-х", "ретро 90-х", "90s",
        "90s minimalism", "90s grunge"
    ],
    "y2k": [
        "y2k", "нулевые", "2000-е", "двухтысячные", "начало 2000-х",
        "начало нулевых", "мода нулевых", "2000-е годы", "нулевые годы",
        "нулевая мода", "nulевые", "cyber y2k", "y2k киберпанк",
        "y2k cyberpunk", "киберпанк 2000-х", "cybery2k"
    ],
    "dark academia": [
        "dark academia", "темная академия", "dark academia",
        "темное academia", "дарк-адепия"
    ],
    "coquette": [
        "coquette", "кокетт", "кокетка", "кокетливый", "кохкет",
        "коquette", "кокетство", "кокетливый стиль", "кокетливый образ",
        "coquette aesthetic"
    ],
    "бохо": [
        "бохо", "богемный", "boho", "boho-chic", "бохо-шик", "богемная",
        "богема", "богемный стиль", "богемная мода", "boho luxe"
    ],
    "романтика": [
        "романтика", "романтический", "романтический стиль", "романтичный",
        "романтическая атмосфера", "романтический образ",
        "романтичное настроение", "романтическая", "романтическая эстетика",
        "романтический ужин", "романтический вечер"
    ],
    "Pinterest": [
        "Pinterest", "Pinterest 2025"
    ],
    "cottagecore": [
        "cottagecore", "коттеджкор", "прибрежная бабушка",
        "прибрежная ковбойка", "coastal cowgirl", "coastal grandmother"
    ],
    "преппи": [
        "преппи", "preppy"
    ],
    "минимализм": ["минимализм", "minimalism", "минималист"]
}

# Create lookup dictionaries
SEASON_LOOKUP = {form: season for season, forms in SEASONS.items() for form in forms}
OCCASION_LOOKUP = {form: occasion for occasion, forms in OCCASIONS.items() for form in forms}
WEATHER_LOOKUP = {form: weather for weather, forms in WEATHER_SYNONYMS.items() for form in forms}
AESTHETIC_LOOKUP = {alias.lower(): k for k, v in AESTHETIC_ALIASES.items() for alias in v}

def preprocess_lookup_forms():
    """Pre-process all lookup forms with spaCy to avoid repeated processing"""
    logging.info("Pre-processing lookup forms...")
    processed_lookups = {}
    
    # Combine all forms for batch processing
    all_forms = []
    form_mapping = {}
    
    # Collect all forms
    for season, forms in SEASONS.items():
        key = f"Сезон:{season}"
        for form in forms:
            all_forms.append(form.lower())
            form_mapping[form.lower()] = key
            
    for occasion, forms in OCCASIONS.items():
        key = f"Случай:{occasion}"
        for form in forms:
            all_forms.append(form.lower())
            form_mapping[form.lower()] = key
            
    for weather, forms in WEATHER_SYNONYMS.items():
        key = f"Погода:{weather}"
        for form in forms:
            all_forms.append(form.lower())
            form_mapping[form.lower()] = key
            
    for aesthetic, aliases in AESTHETIC_ALIASES.items():
        key = f"Эстетика:{aesthetic}"
        for alias in aliases:
            all_forms.append(alias.lower())
            form_mapping[alias.lower()] = key

    # Process all forms in a single batch
    docs = list(nlp.pipe(all_forms))
    
    # Initialize processed lookups
    for key in set(form_mapping.values()):
        processed_lookups[key] = set()
    
    # Add processed forms and their lemmas
    for form, doc in zip(all_forms, docs):
        key = form_mapping[form]
        processed_lookups[key].add(form)
        processed_lookups[key].update(token.lemma_ for token in doc)
    
    logging.info("Lookup forms pre-processing complete")
    return processed_lookups

def process_batch(batch, processed_lookups):
    """Process a batch of entries"""
    results = []
    
    for entry in batch:
        instruction = entry["instruction"]  # Only process the question
        text_lower = instruction.lower()
        
        # Root form of question to deduplicate
        root_key = re.sub(r'[^\w\s]', '', instruction.lower()).strip()
        root_key = " ".join(root_key.split()[:10])
        
        # Process text
        doc = nlp(instruction)
        text_lemmas = {token.lemma_ for token in doc}
        
        # Extract features with context awareness
        features = defaultdict(set)
        
        # Check aesthetics - only when directly mentioned as style
        for aesthetic, forms in AESTHETIC_ALIASES.items():
            for form in forms:
                if form.lower() in text_lower and any(style in text_lower for style in ['стиль', 'стиле', 'образ в']):
                    features["aesthetics"].add(aesthetic)
                    break

        # Check occasions - only when directly asking about or mentioning specific occasion
        for occasion, forms in OCCASIONS.items():
            for form in forms:
                if form.lower() in text_lower and any(prep in text_lower for prep in ['для', 'на', 'в']):
                    features["occasions"].add(occasion)
                    break

        # Check weather - only when asking about weather conditions
        for weather, forms in WEATHER_SYNONYMS.items():
            for form in forms:
                if form.lower() in text_lower and any(w in text_lower for w in ['погода', 'погоде', 'погодные условия']):
                    features["weather"].add(weather)
                    break

        # Check seasons - only when explicitly asking about season
        for season, forms in SEASONS.items():
            for form in forms:
                if form.lower() in text_lower and any(s in text_lower for s in ['сезон', 'время года']):
                    features["seasons"].add(season)
                    break
        
        # Check for trends - only when asking about trends
        if "2024" in text_lower and any(t in text_lower for t in ['тренд', 'модно в', 'актуально в']):
            features["trends"].add("2024")
        if "2025" in text_lower and any(t in text_lower for t in ['тренд', 'модно в', 'актуально в']):
            features["trends"].add("2025")
        
        # Check for combinations - only when explicitly asking about combinations
        combination_patterns = [
            r'(?:как|можно ли|стоит ли).*(?:сочетать|носить|комбинировать)',
            r'(?:с чем|какими).*(?:сочетается|носить|комбинировать)',
            r'что подойдет к',
            r'как дополнить'
        ]
        has_combination = any(re.search(pattern, text_lower) for pattern in combination_patterns)
        
        if any(features.values()) or has_combination:
            results.append({
                "root_key": root_key,
                "entry": {
                    **entry,
                    "features": {
                        "seasons": list(features["seasons"]),
                        "occasions": list(features["occasions"]),
                        "weather": list(features["weather"]),
                        "aesthetics": list(features["aesthetics"]),
                        "trends": list(features["trends"]),
                        "has_combination": has_combination
                    }
                }
            })
    
    return results

def process_fashion_qa():
    """Main function to process and filter fashion QA data"""
    try:
        # Define paths
        input_path = Path("src/data/data/fashion_qa_from_augs.json")
        output_path = Path("src/data/data/filtered_fashion_qa.json")
        intermediate_path = Path("src/data/data/intermediate_fashion_qa.json")

        # Create directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)

        # Load input data
        logging.info(f"Loading data from {input_path}")
        with input_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Pre-process lookups once
        processed_lookups = preprocess_lookup_forms()
        
        # Prepare batches (every 5th entry)
        entries_to_process = [entry for i, entry in enumerate(raw_data) if i % 5 == 0]
        batch_size = 100
        batches = [entries_to_process[i:i + batch_size] 
                  for i in range(0, len(entries_to_process), batch_size)]
        
        logging.info(f"Processing {len(entries_to_process)} entries in {len(batches)} batches")
        
        # Process batches in parallel
        with mp.Pool() as pool:
            process_fn = partial(process_batch, processed_lookups=processed_lookups)
            all_results = []
            seen_roots = set()
            filtered = []
            
            # Process batches with progress bar
            for batch_results in tqdm(pool.imap(process_fn, batches), 
                                    total=len(batches),
                                    desc="Processing batches"):
                # Filter duplicates and collect results
                for result in batch_results:
                    if result["root_key"] not in seen_roots:
                        seen_roots.add(result["root_key"])
                        filtered.append(result["entry"])
                
                # Save intermediate results
                with intermediate_path.open("w", encoding="utf-8") as f:
                    json.dump(filtered, f, ensure_ascii=False, indent=2)

        # Save final results
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "statistics": {
                    "total_entries": len(raw_data),
                    "processed_entries": len(entries_to_process),
                    "filtered_entries": len(filtered),
                    "unique_questions": len(seen_roots)
                },
                "filtered_data": filtered
            }, f, ensure_ascii=False, indent=2)

        logging.info(f"Filtering complete. Processed {len(entries_to_process)} entries, kept {len(filtered)}")

    except Exception as e:
        logging.error(f"Error processing fashion QA data: {str(e)}")
        raise

if __name__ == "__main__":
    process_fashion_qa()
