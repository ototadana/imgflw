# imgflw
A demo application for image editing using LLM.

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
Upload the image you want to edit.

![Upload](./readme-images/step-01.png)

Describe how you want to edit the image in "Input your request here:" and click "Edit".

![Edit](./readme-images/step-02.png)