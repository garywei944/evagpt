import logging
from typing import Any

__all__ = ["setup_root_logger", "log_header"]

logger = logging.getLogger(__name__)


def setup_root_logger(
    thread_info: bool = False, process_info: bool = False, full_path: bool = False
):
    old_factory = logging.getLogRecordFactory()

    def hex_tid_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        # CPython use pthread_t as unsigned long, and the lowest 8 bits are usually 0
        # in normal x64 systems, the address space is usually 48-bit, and the user
        # space starts with 0x7f, so we remove the leading '7f' to make thread id more compact
        record.thread_hex = hex(record.thread)[2:]  # type: ignore[arg-type]

        return record

    if thread_info:
        logging.setLogRecordFactory(hex_tid_factory)

    log_format_segs = ["%(levelname).1s%(asctime)s.%(msecs)03d"]

    if process_info:
        log_format_segs.append("[%(process)d:%(processName)s]")
    if thread_info:
        log_format_segs.append("[%(thread_hex)s:%(threadName)s]")

    log_format_segs.append("%(name)s")
    log_format_segs.append("|")

    location_fmt = (
        "%(pathname)s:%(lineno)d:%(funcName)s"
        if full_path
        else "%(filename)s:%(lineno)d:%(funcName)s"
    )

    log_format_segs.extend([location_fmt, "-", "%(message)s"])

    logging.basicConfig(
        level=logging.DEBUG,
        format=" ".join(log_format_segs),
        datefmt="%m%d %H:%M:%S",
        force=True,
    )


def log_header(header: str, content: Any, header_length: int = 40, log_level: int = logging.DEBUG):
    logger.log(
        log_level,
        "%s %s %s\n\n%s\n",
        "=" * 20,
        header.center(header_length - 2),
        "=" * 30,
        content,
    )
