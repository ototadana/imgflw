# imgflw
A demo application for image editing using LLM.

![demo](./readme-images/demo-01.webp)

## imgflw Application Processing Flow
```mermaid
sequenceDiagram
    participant User
    participant imgflw as User Interface
    participant OpenAI_Embed as OpenAI<br/>Embedding API
    participant OpenAI_TextGen as OpenAI<br/>Text Generation API
    participant Chroma as Chroma<br/>Vector Store
    participant FaceEditor as Image Processing<br/>Components

    User->>imgflw: 1. Input request
    imgflw->>OpenAI_Embed: 2. Convert request to embedding
    imgflw->>Chroma: 3. Search in Chroma
    alt Similar request exists in Chroma
        Chroma-->>imgflw: 4a. Retrieve corresponding workflow definition (JSON)
    else No similar request in Chroma
        imgflw->>OpenAI_TextGen: 4b. Generate new workflow definition (JSON)
    end
    imgflw->>FaceEditor: 5. Process image using workflow definition
```

- [OpenAPI Schema Specification for Workflow](./imgflw/components/core/workflow_generators/workflow.yml)

## Installation and Launch
Clone this repository:

```bash
git clone https://github.com/ototadana/imgflw
cd imgflw
```

Install [PyTorch 2.1](https://pytorch.org/) and [xFormers](https://github.com/facebookresearch/xformers):

Example command:

```bash
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
```

Install required software:

```bash
pip install -r requirements.txt
```

To launch the application:

```bash
python main.py
```

## Initial Setup
Once the interface appears in the browser, switch to the "Settings" tab. Enter your API key in "Workflow Generator - OpenAI API Key" and click "Save".

![OpenAI API Key](./readme-images/settings-01.jpg)

## Editing Images
1. Upload the image you want to edit.
  ![Upload](./readme-images/step-01.png)

2. Describe how you want to edit the image in "Input your request here:" and click "Edit".
  ![Edit](./readme-images/step-02.png)

## License
This software is released under the MIT License, see [LICENSE](./LICENSE).

## Acknowledgements
This application has been developed with the support of several outstanding software resources:

#### Workflow Definition Generation
- [OpenAI Text generation API](https://platform.openai.com/docs/guides/text-generation/text-generation-models)

#### Workflow Definition Storage
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings/embeddings)
- [Chroma](https://docs.trychroma.com/)

#### Image Processing
- [Face Editor](https://github.com/ototadana/sd-face-editor)
- [Diffusers](https://huggingface.co/docs/diffusers/index)
- [facexlib](https://github.com/xinntao/facexlib)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)

#### User Interface
- [Gradio](https://www.gradio.app/)