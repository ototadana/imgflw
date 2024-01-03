from imgflw.io import util as io_util
from imgflw.usecase import WorkflowGenerator


class FileWorkflowGenerator(WorkflowGenerator):
    def generate(self, request_content: str) -> str:
        if request_content.startswith("http"):
            return io_util.fetch(request_content)

        file = io_util.get_asset(f"workflows/{request_content}")
        with open(file, "r") as f:
            return f.read()
