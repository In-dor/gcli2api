"""
日志实时发布通道：支持多订阅者、跨线程安全投递。
"""

import asyncio
import threading
from typing import Dict, Tuple


class LogRealtimeChannel:
    """事件驱动日志发布通道。"""

    def __init__(self, queue_maxsize: int = 1000):
        self._queue_maxsize = queue_maxsize
        self._subscribers: Dict[int, Tuple[asyncio.AbstractEventLoop, asyncio.Queue[str]]] = {}
        self._lock = threading.Lock()
        self._next_subscriber_id = 1

    async def subscribe(self) -> Tuple[int, asyncio.Queue[str]]:
        """在当前事件循环中创建一个订阅队列并返回订阅ID。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._queue_maxsize)

        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            self._subscribers[subscriber_id] = (loop, queue)

        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: int):
        """取消订阅。"""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, message: str):
        """线程安全发布：可从非 asyncio 线程调用。"""
        if not message:
            return

        with self._lock:
            subscribers = list(self._subscribers.items())

        for subscriber_id, (loop, queue) in subscribers:
            try:
                loop.call_soon_threadsafe(self._enqueue_message, subscriber_id, queue, message)
            except RuntimeError:
                # 事件循环已关闭，移除失效订阅
                self.unsubscribe(subscriber_id)

    def _enqueue_message(self, subscriber_id: int, queue: asyncio.Queue[str], message: str):
        """在订阅者所属事件循环中执行入队。"""
        try:
            if queue.full():
                try:
                    queue.get_nowait()  # 丢弃最旧消息，保证最新日志优先
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # 极端并发下仍满则丢弃本条，避免阻塞发布线程
            pass
        except Exception:
            self.unsubscribe(subscriber_id)


log_realtime_channel = LogRealtimeChannel()
