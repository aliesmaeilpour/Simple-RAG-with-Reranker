import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class SessionLock:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class SessionLocks:
    def __init__(self) -> None:
        self._locks: dict[str, SessionLock] = {}

    @asynccontextmanager
    async def acquire(self, session_id: str) -> AsyncIterator[None]:
        entry = self._locks.setdefault(session_id, SessionLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users <= 0 and self._locks.get(session_id) is entry:
                del self._locks[session_id]

    def __len__(self) -> int:
        return len(self._locks)

    def active_sessions(self) -> list[str]:
        return list(self._locks.keys())