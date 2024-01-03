import os
from urllib.parse import urlparse

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torch.hub import download_url_to_file

from imgflw.entities import Image
from imgflw.io import util as io_util
from imgflw.usecase import Settings, Upscaler


class RealESRGANx4plus(Upscaler):
    def name(self) -> str:
        return "RealESRGAN x4+"

    def upscale(self, image: Image) -> Image:
        file_path = self.__load_model(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "models/ESRGAN"
        )
        upscaler = RealESRGANer(
            scale=4,
            model_path=file_path,
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            half=False,
            tile=192,
            tile_pad=8,
            device=Settings.device,
        )

        scaled = upscaler.enhance(image.array, outscale=4)[0]
        return Image(scaled)

    def __load_model(self, url: str, dir: str) -> str:
        dir = io_util.get_asset(dir)
        file_name = os.path.basename(urlparse(url).path)
        file_path = os.path.abspath(os.path.join(dir, file_name))
        if not os.path.exists(file_path):
            os.makedirs(dir, exist_ok=True)
            print(f"Downloading {url} to {file_path}", flush=True)
            download_url_to_file(url, file_path, progress=True)
        return file_path
