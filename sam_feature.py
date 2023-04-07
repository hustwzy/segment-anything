import tyro
import torch
import cv2
import dataclasses
import enum
import pathlib
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide


class ModelTypeEnum(enum.Enum):
    default = enum.auto()
    vit_l = enum.auto()
    vit_b = enum.auto()


@dataclasses.dataclass
class Args:
    model_type: ModelTypeEnum = ModelTypeEnum.default
    """The type of model to load, in ['default', 'vit_l', 'vit_b']"""

    checkpoint: pathlib.Path = pathlib.Path("checkpoint.pth")
    """The path to the SAM checkpoint to use for mask generation."""

    input: pathlib.Path = pathlib.Path("input")
    """Path to either a single input image or folder of images."""

    output: pathlib.Path = pathlib.Path("output")
    """Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single json with COCO-style masks."""

    multimask_output: bool = False
    """Whether to output a single mask per image or multiple masks per image."""

    device: str = "cuda"
    """The device to run generation on."""

    batch_size: int = 1
    """The batch size to use for generation."""


def load_data(file_path: pathlib.Path):
    imgs = []
    if file_path.is_dir():
        for img_path in file_path.iterdir():
            imgs.append(cv2.imread(str(img_path)))
    else:
        imgs.append(cv2.imread(str(file_path)))
    return imgs



def main(args: Args):
    sam: Sam = sam_model_registry[args.model_type.name](checkpoint=args.checkpoint)
    sam.to(args.device)
    imgs = load_data(args.input)
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    for img in imgs:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=args.device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        input_image_torch = sam.preprocess(input_image_torch)
        features = sam.image_encoder(input_image_torch)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)