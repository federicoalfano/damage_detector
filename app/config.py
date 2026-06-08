from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/db.sqlite3"
    api_key: str = ""  # empty = no auth check (local dev)
    # When True, an empty api_key is a hard misconfiguration flagged loudly at
    # boot instead of silently shipping an open instance. Defaults False so local
    # dev keeps working; set REQUIRE_AUTH=true in production.
    require_auth: bool = False
    # Gate diagnostic endpoints (e.g. debug-photos, which leaks server paths).
    debug_endpoints: bool = False
    openai_api_key: str = ""
    openai_base_url: str = ""
    # Validated prod model = google/gemini-2.5-flash via OpenRouter. The old
    # default "o4-mini" was a footgun: unavailable on the OpenRouter base_url AND
    # _is_reasoning_model() => True, which silently drops temperature and switches
    # to max_completion_tokens. Keep a sane Gemini default if the env is unset.
    openai_model: str = "google/gemini-2.5-flash"
    # Independent VLM passes per photo, merged by union (recall-first) with
    # agreement -> confidence. 1 = single pass (legacy). 3 = recommended for
    # critical-damage recall. Cost/latency scale ~linearly with this.
    vlm_passes: int = 3
    # Independent runs of the scudo tiled detail pass, merged union-wise. The
    # tile pass is the ENTIRE source of critical-damage recall and is highly
    # nondeterministic (same photo: 4/10/31 criticals across runs), so a single
    # run randomly misses real damage. 2 = recommended to recover those misses.
    # Only affects scudo (tiling is gated on the reference-image vehicle types).
    tile_passes: int = 2
    data_dir: str = "./data"
    max_photo_size_bytes: int = 2 * 1024 * 1024  # 2MB
    cors_origins: list[str] = ["http://localhost:8000", "http://192.168.1.200:8000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
