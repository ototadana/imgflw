from imgflw.components.core.workflow_generators.openai import OpenAIWorkflowGenerator
from imgflw.components.core.workflow_stores.chroma import ChromaWorkflowStore
from imgflw.ui import UIBuilder
from imgflw.usecase import Settings

Settings.load()
ui_builder = UIBuilder(OpenAIWorkflowGenerator(), ChromaWorkflowStore())
ui = ui_builder.build()

if __name__ == "__main__":
    ui.queue().launch(inbrowser=True)
