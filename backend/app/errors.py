"""One error shape for the whole API: {"error": {"code": "...", "message": "..."}}.

Messages are safe to show to users. Internal details (stack traces, keys, provider responses)
never go into them.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.schemas import ErrorDetail, ErrorResponse


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ClientDisconnected(Exception):
    """Phase 25: the browser closed a streamed answer. Raised inside the pipeline to stop it; never shown."""


def error_response(code: str, message: str, status_code: int) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return error_response(exc.code, exc.message, exc.status_code)

    @app.exception_handler(SQLAlchemyError)
    async def handle_database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        # e.g. saving conversation history while the database is down; details stay out of the response
        return error_response("DATABASE_UNAVAILABLE", "The database is unavailable right now.", 503)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = exc.errors()
        if any(err.get("type") == "json_invalid" for err in errors):
            return error_response("INVALID_JSON", "Request body is not valid JSON.", 400)
        fields = {str(err["loc"][-1]) for err in errors if err.get("loc")}
        if "conversation_id" in fields:
            message = "conversation_id must be a UUID."
        else:
            message = "Message must be between 1 and 2000 characters."
        return error_response("INVALID_INPUT", message, 422)
