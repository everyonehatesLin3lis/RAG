"""One error shape for the whole API: {"error": {"code": "...", "message": "..."}}.

Messages are safe to show to users. Internal details (stack traces, keys, provider responses)
never go into them.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorDetail, ErrorResponse


logger = logging.getLogger(__name__)

HTTP_ERRORS = {  # Phase 27: FastAPI's own errors in our shape instead of {"detail": "..."}
    404: ("NOT_FOUND", "There is no such endpoint."),
    405: ("METHOD_NOT_ALLOWED", "This endpoint does not accept that method."),
}


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
    """Call before adding CORSMiddleware: the catch-all below must sit inside it, so even a crash response carries the
    CORS headers and the browser can read the error instead of reporting that the server cannot be reached."""

    @app.middleware("http")
    async def catch_unexpected_errors(request: Request, call_next):
        # Phase 27: a bug anywhere would otherwise answer with Starlette's plain-text "Internal Server Error".
        try:
            return await call_next(request)
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return error_response("INTERNAL_ERROR", "Something went wrong on the server.", 500)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = HTTP_ERRORS.get(exc.status_code, ("HTTP_ERROR", "The request could not be handled."))
        return error_response(code, message, exc.status_code)

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
