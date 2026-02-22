from typing import Any


def success_response(data: Any = None, message: str | None = None) -> dict:
    return {"status": "success", "data": data, "message": message}


def error_response(message: str, data: Any = None) -> dict:
    return {"status": "error", "data": data, "message": message}
