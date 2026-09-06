"""Terminal rendering helpers for the IPA orchestrator."""
from __future__ import annotations

from . import orchestration as _orchestration

globals().update({name: value for name, value in vars(_orchestration).items() if not name.startswith("__")})

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    GRAY    = "\033[90m"
    BG_BLUE = "\033[44m"

    @staticmethod
    def status_icon(health: str) -> str:
        return {
            "healthy": f"{C.GREEN}●{C.RESET}",
            "stuck":   f"{C.YELLOW}●{C.RESET}",
            "dead":    f"{C.RED}●{C.RESET}",
            "done":    f"{C.GREEN}✓{C.RESET}",
            "error":   f"{C.RED}✗{C.RESET}",
            "paused":  f"{C.YELLOW}⏸{C.RESET}",
        }.get(health, f"{C.GRAY}○{C.RESET}")


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def fmt_time(seconds: float) -> str:
    if seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return f"[{'?' * width}]"
    pct = min(current / total, 1.0)
    filled = int(pct * width)
    bar = f"{C.CYAN}{'█' * filled}{C.GRAY}{'░' * (width - filled)}{C.RESET}"
    pct_str = f"{pct * 100:5.1f}%"
    return f"[{bar}] {pct_str}"


