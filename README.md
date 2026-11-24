# Как люди ищут путь в Википедии (Wikispeedia)

Учебный исследовательский проект по анализу того, как люди ищут путь в графе статей Википедии, чем их маршруты отличаются от кратчайших и можно ли предсказывать их действия с помощью ML. В основе работы лежит исследование [West & Leskovec (WWW 2012)](https://snap.stanford.edu/data/wikispeedia.html).

## Данные

Используется датасет **Wikispeedia** из коллекции [Stanford SNAP](https://snap.stanford.edu/data/wikispeedia.html).

## Установка и запуск

1. Создание виртуального окружения и установка зависимостей:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # на Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Загрузка и распаковка сырых данных SNAP:
   ```bash
   python scripts/download_dataset.py
   ```

3. Предобработка и построение общих артефактов:
   ```bash
   python scripts/build_dataset.py
   ```
