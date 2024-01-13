import os
from string import Template

from openai import OpenAI

from imgflw.usecase import Settings, WorkflowGenerator


class OpenAIWorkflowGenerator(WorkflowGenerator):
    def __init__(self):
        self.client = OpenAI(api_key="...")
        self.system_message = ""

    def generate(self, request: str) -> str:
        self.client.api_key = Settings.get("openai_api_key")

        response = self.client.chat.completions.create(
            messages=[self.get_system_message(), {"role": "user", "content": request}],
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        return content

    def get_system_message(self) -> str:
        if not self.system_message:
            instruction = Template(self.read_file("instruction-for-generation.txt"))
            workflow_spec = self.read_file("workflow.yml")
            system_message = instruction.substitute(workflow_spec=workflow_spec)
            self.system_message = {"role": "system", "content": system_message}
        return self.system_message

    def read_file(self, file_name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        with open(path, "r") as f:
            return f.read()
