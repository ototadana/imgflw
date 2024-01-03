from pydantic import BaseModel


class Config(BaseModel):
    model: str = "stabilityai/sd-turbo"
    prompt: str = ""
    negative_prompt: str = ""
    max_face_count: int = 20
    show_intermediate_steps: bool = True
    img2img_size: int = 512
    use_minimal_area: bool = False
    face_margin: float = 1.6
    seed: int = 2
