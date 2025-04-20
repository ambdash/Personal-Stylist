# Personal-Stylist


project-ai-stylist/
├── .github/                       # Конфигурация GitHub Actions, шаблоны для Issues и PR
│   └── workflows/
│       └── ci.yml                 # Пример CI/CD для запуска тестов, линтинга и сборки Docker
├── docker/                        # Файлы для контейнеризации
│   ├── Dockerfile.api             # Dockerfile для FastAPI
│   ├── Dockerfile.neo4j           # Dockerfile для Neo4j (при необходимости кастомизации)
│   ├── docker-compose.yml         # Сборка контейнеров (API, Neo4j, брокер сообщений, мониторинг)
├── docs/                          # Документация проекта
│   ├── architecture.md          # Описание архитектуры и компонентов системы
│   └── api_documentation.md       # Документация по API (эндпоинты /recommend, /train и др.)
├── data/                          # Данные и DVC-файлы
│   ├── raw/                       # Исходные данные (например, выгрузки с Pinterest, Vogue)
│   ├── processed/                 # Обработанные данные для обучения
│   └── dvc.yaml                   # DVC pipeline для версионирования данных
├── infra/                         # Инфраструктурные скрипты и конфигурации
│   ├── neo4j/                     # Скрипты для развёртывания и инициализации графовой БД
│   ├── messaging/                 # Конфигурация для брокера сообщений (Kafka/RabbitMQ)
│   └── monitoring/                # Скрипты и конфигурации для Prometheus, Grafana, OpenTelemetry
├── src/                           # Исходный код проекта
│   ├── api/                       # FastAPI сервис
│   │   ├── main.py                # Точка входа FastAPI
│   │   ├── endpoints/             # Эндпоинты, например, /recommend и /train
│   │   └── config.py              # Конфигурация приложения
│   ├── bot/                       # Telegram-бот
│   │   ├── bot.py                 # Основная логика Telegram-бота
│   │   └── handlers/              # Обработчики сообщений и команд
│   ├── models/                    # Модели, скрипты для обучения и инференса LLM
│   │   ├── finetune.py            # Файнтюнинг модели с использованием LoRA
│   │   └── inference.py           # Запуск inference-сервиса
│   ├── rag/                       # Модуль RAG для интеграции с графовой БД и поиска
│   │   └── search.py              # Логика поиска по модным статьям и стилям
│   └── utils/                     # Вспомогательные функции, логирование, обработка ошибок
├── tests/                         # Тесты (pytest)
│   ├── test_api.py                # Тесты для FastAPI эндпоинтов
│   ├── test_bot.py                # Тесты для Telegram-бота
│   └── test_model.py              # Тесты для LLM и RAG компонентов
├── .env                           # Переменные окружения (не коммитить в git)
├── .gitignore                     # Файлы/папки, игнорируемые Git (venv, .env, __pycache__, т.д.)
├── pyproject.toml                 # Конфигурация Poetry (зависимости, скрипты, версии)
├── README.md                      # Общее описание проекта, инструкции по установке и запуску
└── requirements.txt               # Если используется pip (можно генерировать из Poetry)``