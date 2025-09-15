from abc import ABC, abstractmethod
from typing import List, Dict
from queue import Queue
from ..communication.message import Message


class BaseAgent(ABC):
    def __init__(self):
        self.message_queue = Queue()

    @abstractmethod
    async def process_message(self, message: Message):
        pass
