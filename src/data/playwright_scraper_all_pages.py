
import asyncio
from playwright.async_api import async_playwright
import json
import os

OUTPUT_DIR = "elyts_articles_html"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_URL = "https://elyts.ru"
START_URL = "https://elyts.ru/blog/s/modnye-sovety/"
PAGE_URL_TEMPLATE = "https://elyts.ru/blog/s/modnye-sovety/?PAGEN_1={}"

async def fetch_article_links(page):
    all_links = set()
    MAX_PAGES = 20
    page_number = 1

    while page_number <= MAX_PAGES:
        page_url = START_URL if page_number == 1 else PAGE_URL_TEMPLATE.format(page_number)
        print(f"Обрабатывается страница: {page_url}")
        await page.goto(page_url)
        await page.wait_for_load_state("domcontentloaded")

        links = await page.eval_on_selector_all(
            "div.product-item-container a[href^='/blog/']",
            "elements => elements.map(el => el.href)"
        )

        # Добавляем новые
        initial_count = len(all_links)
        all_links.update(links)

        # Проверка: если не прибавилось — выходим
        if len(all_links) == initial_count:
            print("Больше новых ссылок не найдено. Остановка.")
            break

        page_number += 1

    return sorted(all_links)

async def save_article(page, url, index):
    await page.goto(url)
    await page.wait_for_load_state("domcontentloaded")
    html = await page.content()

    filename = os.path.join(OUTPUT_DIR, f"article_{index:03}.html")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Сохранено: {filename}")
    return {"id": index, "url": url, "file": filename}

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("Получаем список статей со всех страниц...")
        links = await fetch_article_links(page)
        print(f"Найдено {len(links)} статей.")

        log = []
        for i, link in enumerate(links, start=1):
            try:
                result = await save_article(page, link, i)
                log.append(result)
            except Exception as e:
                print(f"Ошибка при обработке {link}: {e}")

        await browser.close()

        with open("elyts_articles_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print("Лог сохранён: elyts_articles_log.json")

if __name__ == "__main__":
    asyncio.run(main())
