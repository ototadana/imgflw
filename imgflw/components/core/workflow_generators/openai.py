import os

from openai import OpenAI

from imgflw.usecase import Settings, WorkflowGenerator


class OpenAIWorkflowGenerator(WorkflowGenerator):
    def __init__(self):
        self.client = OpenAI(api_key="...")
        self.system_message = ""

    def generate(self, request: str) -> str:
        self.client.api_key = Settings.get("openai_api_key")

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            seed=2048,
            messages=[self.get_system_message(), {"role": "user", "content": request}],
            temperature=0,
            max_tokens=256,
            top_p=0.4,
            frequency_penalty=0,
            presence_penalty=0,
        )

        content = response.choices[0].message.content
        return content

    def get_system_message(self) -> str:
        if not self.system_message:
            instruction = self.read_file("generate-instruction.txt")
            workflow_spec = self.read_file("workflow.yml")
            system_message = instruction.format(workflow_spec=workflow_spec)
            self.system_message = {"role": "system", "content": system_message}
        return self.system_message

    def read_file(self, file_name: str):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
        with open(path, "r") as f:
            return f.read()
