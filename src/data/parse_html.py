from lxml import html
import os
import jsonlines

INPUT_DIR = "elyts_articles_html"
OUTPUT_FILE = "articles.jsonl"

def extract_article_from_file(filepath):
    with open(filepath, encoding="utf-8") as f:
        tree = html.fromstring(f.read())

    # Заголовок статьи
    title = tree.xpath('//title/text()')
    title = title[0].strip() if title else "Без названия"

    # Основной текст статьи
    paragraphs = tree.xpath('//div[@class="blog-detail"]//p[@class="js-text"]/text()')
    headings = tree.xpath('//div[@class="blog-detail"]//h2/text() | //div[@class="blog-detail"]//h3/text()')
    
    content = []
    
    # Сначала добавляем все заголовки и параграфы
    for element in tree.xpath('//div[@class="blog-detail"]//*[self::h2 or self::h3 or self::p[@class="js-text"]]'):
        if element.tag in ['h2', 'h3']:
            content.append(f"\n{element.text_content().strip()}\n")
        elif element.tag == 'p' and 'js-text' in element.classes:
            content.append(element.text_content().strip())
    
    # Объединяем все части текста
    text = "\n".join([p for p in content if p.strip()])
    
    return {
        "title": title,
        "text": text,
        "filename": os.path.basename(filepath)
    }

def main():
    with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
        for filename in sorted(os.listdir(INPUT_DIR)):
            if filename.endswith(".html"):
                path = os.path.join(INPUT_DIR, filename)
                article = extract_article_from_file(path)
                if article and article['text'].strip():
                    writer.write(article)
                    print(f"Parsed: {filename}")
                else:
                    print(f"Пропущено (не найдена статья): {filename}")

if __name__ == "__main__":
    main()