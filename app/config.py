from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/db.sqlite3"
    api_key: str = ""  # empty = no auth check (local dev)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    data_dir: str = "./data"
    max_photo_size_bytes: int = 2 * 1024 * 1024  # 2MB
    cors_origins: list[str] = ["http://localhost:8000", "http://192.168.1.200:8000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
