from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import os
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, timedelta
import uuid
from PIL import Image
from io import BytesIO
import fitz
import asyncio
import re

load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERROR_CODES = {
    "TOKEN_ERROR": "Ошибка получения токена доступа",
    "GENERATION_ERROR": "Ошибка генерации описания",
    "FILE_ERROR": "Ошибка обработки файла",
    "PARSE_ERROR": "Ошибка парсинга таблицы",
    "INVALID_REQUEST": "Некорректный запрос",
}

client_id = os.getenv("GIGACHAT_CLIENT_ID")
client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
auth_key = os.getenv("GIGACHAT_AUTH_KEY")

app = FastAPI(
    title="Water Quality Parser API v2",
    description="Улучшенный API для парсинга данных о качестве воды",
    version="2.1.0"
)


class GigaChatClient:
    def __init__(self, client_id: str, client_secret: str, auth_key: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_key = auth_key
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.access_token = None
        self.token_expires_at = None
        self.token_lock = asyncio.Lock()

    async def get_access_token(self) -> str:
        async with self.token_lock:
            if self.access_token and self.token_expires_at:
                if datetime.now() < self.token_expires_at - timedelta(minutes=5):
                    return self.access_token

            try:
                async with httpx.AsyncClient(verify=False) as client:
                    response = await client.post(
                        "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                        headers={
                            "Authorization": f"Basic {self.auth_key}",
                            "RqUID": str(uuid.uuid4()),
                            "Content-Type": "application/x-www-form-urlencoded"
                        },
                        data={"scope": "GIGACHAT_API_PERS"}
                    )

                    if response.status_code != 200:
                        raise HTTPException(
                            status_code=500,
                            detail={
                                "error_code": "TOKEN_ERROR",
                                "error_description": ERROR_CODES["TOKEN_ERROR"],
                                "details": response.text
                            }
                        )

                    data = response.json()
                    self.access_token = data["access_token"]
                    expires_in = data.get("expires_at", 1800000)
                    self.token_expires_at = datetime.now() + timedelta(milliseconds=expires_in)
                    return self.access_token

            except Exception as e:
                logger.error(f"Token error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error_code": "TOKEN_ERROR",
                        "error_description": ERROR_CODES["TOKEN_ERROR"],
                        "details": str(e)
                    }
                )


# Конфигурация столбцов с альтернативными названиями
WATER_COLUMNS = {
    "fk_fp": {
        "name": "ФК (ФП)",
        "aliases": ["ФК (ФП)", "ФК(ФП)", "ФК", "ФП"],
        "unit": "мг/дм³",
        "description": "Фактическая концентрация (фоновый показатель)"
    },
    "dk_star": {
        "name": "ДК*",
        "aliases": ["ДК*", "ДК *"],
        "unit": "мг/дм³",
        "description": "Допустимая концентрация (норматив 1)"
    },
    "fk_dk_star": {
        "name": "ФК/ДК*",
        "aliases": ["ФК/ДК*", "ФК / ДК*"],
        "unit": "",
        "description": "Отношение фактической к допустимой концентрации"
    },
    "dk_double_star": {
        "name": "ДК**",
        "aliases": ["ДК**", "ДК **"],
        "unit": "мг/дм³",
        "description": "Допустимая концентрация (норматив 2)"
    },
    "fk_dk_double_star": {
        "name": "ФК/ДК**",
        "aliases": ["ФК/ДК**", "ФК / ДК**"],
        "unit": "",
        "description": "Отношение фактической к допустимой концентрации (норматив 2)"
    },
    "fk_decl": {
        "name": "ФКдекл (ФПдекл)",
        "aliases": ["ФКдекл", "ФПдекл", "ФКдекл (ФПдекл)"],
        "unit": "мг/дм³",
        "description": "Декларируемая фактическая концентрация"
    },
    "fk_fk_decl": {
        "name": "ФК/ФКдекл",
        "aliases": ["ФК/ФКдекл", "ФК / ФКдекл"],
        "unit": "",
        "description": "Отношение фактической к декларируемой концентрации"
    }
}

# Стандартные материалы
WATER_MATERIALS = [
    "Азот общий",
    "Алюминий",
    "БПК 5",
    "Водородный показатель",
    "Взвешенные вещества",
    "Железо",
    "Жиры",
    "Кадмий",
    "Марганец",
    "Медь",
    "Нефтепродукты",
    "Никель",
    "Свинец",
    "СПАВ (анионные)",
    "СПАВ (неионогенные)",
    "Фенолы",
    "Фосфор общий",
    "ХПК",
    "ХПК/БПК5",
    "Цинк",
    "Сульфаты",
    "Хлориды",
    "Аммоний-ион",
    "Нитриты",
    "Нитраты",
    "Фосфаты"
]

def normalize_material_name(name: str) -> str:
    """Нормализует название материала для сопоставления"""
    normalized = re.sub(r'\s+', ' ', name.strip().lower())
    normalized = re.sub(r'\([^)]*\)', '', normalized).strip()
    return normalized


def create_material_mapping(materials: List[str]) -> Dict[str, str]:
    """Создает маппинг нормализованных названий к оригинальным"""
    mapping = {}
    for material in materials:
        normalized = normalize_material_name(material)
        mapping[normalized] = material
    return mapping


async def process_file_to_image(file: UploadFile, crop_left_half: bool = False) -> tuple[bytes, str]:
    """Обрабатывает файл (PDF или изображение) и возвращает изображение"""
    filename_lower = file.filename.lower()
    is_pdf = filename_lower.endswith('.pdf')

    file_bytes = await file.read()

    try:
        if is_pdf:
            # Конвертируем PDF в изображение
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            page = pdf_document[0]

            # Высокое разрешение для лучшего распознавания
            mat = fitz.Matrix(3.0, 3.0)  # Увеличили с 2.0 до 3.0
            pix = page.get_pixmap(matrix=mat)

            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf_document.close()
        else:
            # Открываем изображение
            image = Image.open(BytesIO(file_bytes))
            # Конвертируем в RGB если нужно
            if image.mode != 'RGB':
                image = image.convert('RGB')

        # Обрезаем если нужно
        if crop_left_half:
            width, height = image.size
            image = image.crop((0, 0, width // 2, height))

        # Сохраняем как PNG
        output_buffer = BytesIO()
        image.save(output_buffer, format='PNG', quality=95, optimize=True)
        output_buffer.seek(0)

        return output_buffer.getvalue(), 'image/png'

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "FILE_ERROR",
                "error_description": f"Ошибка обработки файла: {str(e)}"
            }
        )


def create_enhanced_prompt(materials: List[str], column_info: dict) -> str:
    """Создает улучшенный промпт с четкими инструкциями"""

    materials_list = "\n".join([f"- {mat}" for mat in materials])
    column_aliases = ", ".join(column_info['aliases'])

    prompt = f"""Ты - эксперт по анализу таблиц с данными о качестве воды.

ЗАДАЧА:
Найди в таблице столбец с названием "{column_info['name']}" (может быть указан как: {column_aliases})
и извлеки числовые значения для следующих веществ:

{materials_list}

ИНСТРУКЦИИ:
1. Внимательно изучи заголовки таблицы
2. Найди столбец "{column_info['name']}" - он может быть написан с пробелами, без пробелов, или в скобках
3. Для каждого вещества из списка найди строку в первом столбце таблицы
4. Извлеки ТОЛЬКО ЧИСЛОВОЕ значение из найденного столбца (без единиц измерения)
5. Названия веществ могут быть с разными регистрами, пробелами или скобками - будь гибким при поиске
6. Если значение не указано, стоит прочерк "-" или ячейка пустая - верни null
7. Преобразуй все числа в формат number (int или float), не строки

ВАЖНО:
- Игнорируй все другие столбцы
- Будь точным с числами - не округляй, не меняй значения
- Если в ячейке несколько чисел, бери первое основное значение
- Десятичный разделитель может быть точкой или запятой

Верни результат через функцию extract_water_values."""

    return prompt


@app.post("/api/v1/parse/water")
async def parse_water_quality(
        file: UploadFile = File(..., description="PDF файл или изображение с таблицей"),
        materials: Optional[str] = Form(None, description="Список материалов (через запятую или JSON)"),
        column: str = Form("fk_fp", description="Код столбца для извлечения"),
        crop_left_half: bool = Form(True, description="Обрезать левую половину"),
        max_retries: int = Form(2, description="Максимум попыток запроса")
):
    """
    Парсит таблицу качества воды с улучшенной точностью

    Примеры:

    ```bash
    # Базовый запрос
    curl -X POST "http://localhost:8082/api/v1/parse/water" \\
      -F "file=@water_report.pdf" \\
      -F "column=fk_fp"

    # С указанием материалов
    curl -X POST "http://localhost:8082/api/v1/parse/water" \\
      -F "file=@water_report.pdf" \\
      -F "materials=Алюминий,Железо,БПК5" \\
      -F "column=fk_dk_star"

    # С обрезкой изображения
    curl -X POST "http://localhost:8082/api/v1/parse/water" \\
      -F "file=@table.jpg" \\
      -F "column=fk_fp" \\
      -F "crop_left_half=true"
    ```
    """

    # Проверка столбца
    if column not in WATER_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный столбец '{column}'. Доступные: {', '.join(WATER_COLUMNS.keys())}"
        )

    column_info = WATER_COLUMNS[column]

    # Парсинг списка материалов для фильтрации результата
    requested_materials = None  # None означает вернуть все
    if materials:
        materials = materials.strip()
        if materials.startswith('[') and materials.endswith(']'):
            try:
                requested_materials = json.loads(materials)
                if not isinstance(requested_materials, list):
                    raise ValueError("materials должен быть массивом")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Неверный JSON: {str(e)}")
        else:
            requested_materials = [m.strip() for m in materials.split(',') if m.strip()]

        if not requested_materials:
            raise HTTPException(status_code=400, detail="Список материалов пуст")

    # Инициализация клиента
    gigachat = GigaChatClient(client_id, client_secret, auth_key)
    token = await gigachat.get_access_token()

    # Обработка файла
    image_bytes, content_type = await process_file_to_image(file, crop_left_half)

    # Загрузка в GigaChat
    async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
        upload_resp = await client.post(
            f"{gigachat.base_url}/files",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (f"processed_{file.filename}.png", image_bytes, content_type)},
            data={"purpose": "general"}
        )

    if upload_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки: {upload_resp.text}")

    file_id = upload_resp.json()["id"]

    # Создание маппинга для ВСЕХ материалов (для промпта)
    material_mapping = create_material_mapping(WATER_MATERIALS)

    # Формирование схемы функции со ВСЕМИ материалами
    properties = {}
    for normalized_name in material_mapping.keys():
        safe_key = normalized_name.replace(" ", "_").replace("/", "_")
        original_name = material_mapping[normalized_name]
        properties[safe_key] = {
            "type": "number",
            "description": f"Значение для '{original_name}' в столбце '{column_info['name']}'. Если не найдено - не включай в ответ."
        }

    function_schema = {
        "name": "extract_water_values",
        "description": f"Извлекает числовые значения из столбца '{column_info['name']}'",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": []  # Все поля опциональны
        }
    }

    logger.info(f"Function schema: {json.dumps(function_schema, ensure_ascii=False, indent=2)}")

    # Создание промпта со ВСЕМИ материалами
    prompt = create_enhanced_prompt(WATER_MATERIALS, column_info)

    messages = [{
        "role": "user",
        "content": prompt,
        "attachments": [file_id]
    }]

    # Функция запроса с retry логикой
    async def run_request_with_retry(attempt: int = 1):
        payload = {
            "model": "GigaChat-Pro",
            "messages": messages,
            "functions": [function_schema],
            "function_call": {"name": "extract_water_values"},
            "stream": False,
            "temperature": 0.1  # Низкая температура для точности
        }

        async with httpx.AsyncClient(verify=False, timeout=180.0) as client:
            response = await client.post(
                f"{gigachat.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

        if response.status_code != 200:
            logger.error(f"Attempt {attempt} failed: {response.text}")
            raise HTTPException(status_code=500, detail=f"Ошибка GigaChat: {response.text}")

        return response.json()

    # Извлечение и валидация результатов
    def extract_and_validate(response_data):
        try:
            args = response_data["choices"][0]["message"]["function_call"]["arguments"]
            logger.info(f"Raw arguments: {args}")

            # Инициализируем все поля как None
            parsed = {}
            all_keys = [normalized_name.replace(" ", "_").replace("/", "_")
                        for normalized_name in material_mapping.keys()]

            for key in all_keys:
                parsed[key] = None

            # Обновляем найденные значения
            for key, value in args.items():
                if value is None or value == "null" or value == "" or value == "-":
                    parsed[key] = None
                else:
                    try:
                        # Обрабатываем запятые как десятичный разделитель
                        if isinstance(value, str):
                            value = value.replace(',', '.').strip()
                        parsed[key] = float(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Cannot convert {key}={value}: {e}")
                        parsed[key] = None

            return parsed

        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Extraction error: {e}")
            return None

    # Попытки запроса
    result_data = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries}")

        try:
            response = await run_request_with_retry(attempt)
            logger.info(f"Response: {json.dumps(response, ensure_ascii=False)}")

            result_data = extract_and_validate(response)

            if result_data is not None:
                # Проверяем что хоть что-то найдено
                non_null_count = sum(1 for v in result_data.values() if v is not None)
                if non_null_count > 0:
                    logger.info(f"Success on attempt {attempt}, found {non_null_count} values")
                    break
                else:
                    logger.warning(f"Attempt {attempt}: all values are null")

            if attempt < max_retries:
                await asyncio.sleep(2)  # Пауза перед повтором

        except Exception as e:
            logger.error(f"Attempt {attempt} exception: {e}")
            if attempt == max_retries:
                raise

    if result_data is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": "PARSE_ERROR",
                "error_description": "Не удалось извлечь данные из таблицы",
                "suggestion": "Проверьте качество изображения и наличие таблицы в файле"
            }
        )

    # Маппинг обратно к оригинальным названиям (все материалы)
    all_readable_values = {}
    for normalized_name, original_name in material_mapping.items():
        safe_key = normalized_name.replace(" ", "_").replace("/", "_")
        all_readable_values[original_name] = result_data.get(safe_key)
    print(all_readable_values)
    # Фильтрация по запрошенным материалам
    if requested_materials is not None:
        # Возвращаем только запрошенные материалы
        readable_values = {}
        for material in requested_materials:
            readable_values[material] = all_readable_values.get(material)

        # Статистика только по запрошенным
        found_count = sum(1 for v in readable_values.values() if v is not None)
        materials_count = len(requested_materials)
    else:
        # Возвращаем все материалы
        readable_values = all_readable_values
        found_count = sum(1 for v in readable_values.values() if v is not None)
        materials_count = len(WATER_MATERIALS)

    return {
        "status": "success",
        "file_type": "pdf" if file.filename.lower().endswith('.pdf') else "image",
        "column": column,
        "column_name": column_info["name"],
        "materials_requested": materials_count,
        "materials_found": found_count,
        "values": readable_values,
        "metadata": {
            "cropped": crop_left_half,
            "attempts_made": attempt,
            "extracted_at": datetime.now().isoformat(),
            "filtered": requested_materials is not None
        }
    }
@app.get("/api/v1/info/columns")
async def get_available_columns():
    """Получить информацию о доступных столбцах"""
    return {
        "columns": {
            code: {
                "name": info["name"],
                "unit": info["unit"],
                "description": info["description"],
                "aliases": info["aliases"]
            }
            for code, info in WATER_COLUMNS.items()
        }
    }


@app.get("/api/v1/info/materials")
async def get_available_materials():
    """Получить список стандартных материалов"""
    return {
        "materials": WATER_MATERIALS,
        "count": len(WATER_MATERIALS)
    }


@app.post("/api/v1/image/upload_and_prompt")
async def upload_image_and_prompt(
        file: UploadFile = File(...),
        prompt: str = Form(...)
):
    """Загрузить изображение и отправить произвольный промпт"""
    gigachat = GigaChatClient(client_id, client_secret, auth_key)
    token = await gigachat.get_access_token()

    # Загрузка файла
    files = {"file": (file.filename, await file.read(), file.content_type)}
    data = {"purpose": "general"}

    async with httpx.AsyncClient(verify=False, timeout=60.0) as client:
        resp = await client.post(
            f"{gigachat.base_url}/files",
            headers={"Authorization": f"Bearer {token}"},
            files=files,
            data=data
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Upload failed: {resp.text}")

    file_id = resp.json()["id"]

    # Генерация ответа
    messages = [{"role": "user", "content": prompt, "attachments": [file_id]}]

    async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
        gen_resp = await client.post(
            f"{gigachat.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "GigaChat-Pro",
                "messages": messages,
                "stream": False
            }
        )

    if gen_resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Generation failed: {gen_resp.text}")

    content = gen_resp.json()["choices"][0]["message"]["content"]
    return {"response": content}




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8082)
