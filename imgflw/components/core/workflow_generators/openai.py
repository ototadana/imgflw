import os
from collections import deque
from typing import Any, Deque, Dict, List

from openai import OpenAI

from imgflw.usecase import Settings, WorkflowGenerator


class RequestResponse:
    def __init__(self, request: str) -> None:
        self.tokens = 0
        self.request = {"role": "user", "content": request}
        self.response = None

    def set_response(self, role: str, content: str) -> None:
        self.response = {"role": role, "content": content}

    def get_messages(self) -> List[Dict[str, Any]]:
        return [self.request, self.response] if self.response else [self.request]


class MessageManager:
    max_tokens = 12000

    def __init__(self):
        system_message_text_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system-message.txt")
        with open(system_message_text_file, "r") as f:
            self.system_message = {"role": "system", "content": f.read()}
            self.system_message_tokens = 6397
            self.total_tokens = self.system_message_tokens
            self.request_responses: Deque[RequestResponse] = deque()

    def add_request(self, request_content: str):
        self.req_res = RequestResponse(request_content)
        self.request_responses.append(self.req_res)

    def add_response(self, role: str, content: str, tokens: int):
        self.req_res.set_response(role, content)
        self.req_res.tokens = tokens - self.total_tokens

        self.total_tokens += self.req_res.tokens
        while self.total_tokens > self.max_tokens:
            self.total_tokens -= self.request_responses.popleft().tokens

    def get_messages(self) -> List[Dict[str, Any]]:
        messages = [self.system_message]
        for rr in self.request_responses:
            messages.extend(rr.get_messages())
        return messages


class OpenAIWorkflowGenerator(WorkflowGenerator):
    def __init__(self):
        self.client = OpenAI(api_key="...")
        self.message_manager = MessageManager()

    def generate(self, request_content: str) -> str:
        self.message_manager.add_request(request_content)
        self.client.api_key = Settings.get("openai_api_key")

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            seed=2048,
            messages=self.message_manager.get_messages(),
            temperature=0,
            max_tokens=256,
            top_p=0.4,
            frequency_penalty=0,
            presence_penalty=0,
        )

        role = response.choices[0].message.role
        content = response.choices[0].message.content
        self.message_manager.add_response(role, content, response.usage.total_tokens)
        return content
