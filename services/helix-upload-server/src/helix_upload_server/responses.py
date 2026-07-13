from __future__ import annotations

from fastapi.responses import JSONResponse


def error_response(status_code: int, error: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": error,
            "detail": detail,
        },
    )


def success_response(data: dict[str, object]) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content={"ok": True, **data},
    )
