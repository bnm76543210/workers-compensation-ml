"""
Утилита логирования. Управляется через src/config.py → DEBUG.
Пишет в консоль (stdout), где запущен `streamlit run`.

Использование в любой странице:
    from src.logger import log
    log("Данные загружены", rows=len(df), cols=df.shape[1])
    log("Ошибка предсказания", level="ERROR", error=str(e))
"""
import logging
import sys

_logger = logging.getLogger("workers_comp")

if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
    ))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False


def log(msg: str, level: str = "INFO", **kwargs) -> None:
    """
    Вывести сообщение в консоль, если DEBUG=True.

    Args:
        msg:    Текст сообщения.
        level:  Уровень: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        **kwargs: Дополнительные поля — выводятся как key=value.
    """
    try:
        from src.config import DEBUG
    except ImportError:
        try:
            from config import DEBUG
        except ImportError:
            DEBUG = False

    if not DEBUG:
        return

    if kwargs:
        extras = "  |  ".join(f"{k}={v}" for k, v in kwargs.items())
        full_msg = f"{msg}  ||  {extras}"
    else:
        full_msg = msg

    getattr(_logger, level.lower(), _logger.info)(full_msg)
