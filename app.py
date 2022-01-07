import os

os.system("pip install gradio==2.4.6")
import torch
import gradio as gr
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
from network.Transformer import Transformer

LOAD_SIZE = 1280
STYLE = "shinkai_makoto"
MODEL_PATH = "models"
COLOUR_MODEL = "RGB"

model = Transformer()
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f"{STYLE}.pth")))
model.eval()

disable_gpu = True


def inference(img):
    # load image
    input_image = img.convert(COLOUR_MODEL)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image

    if disable_gpu:
        input_image = Variable(input_image).float()
    else:
        input_image = Variable(input_image).cuda()

    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5

    return output_image


title = "Anime Background GAN"
description = "Gradio Demo for CartoonGAN by Chen Et. Al. Models are Shinkai Makoto, Hosoda Mamoru, Kon Satoshi, and Miyazaki Hayao."
article = "<p style='text-align: center'><a href='http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf' target='_blank'>CartoonGAN from Chen et.al</a></p><p style='text-align: center'><a href='https://github.com/venture-anime/cartoongan-pytorch' target='_blank'>Github Repo</a></p><p style='text-align: center'><a href='https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch' target='_blank'>Original Implementation from Yijunmaverick</a></p><center><img src='https://visitor-badge.glitch.me/badge?page_id=akiyamasho' alt='visitor badge'></center></p>"

examples = [
    ["examples/garden_in.jpeg", "examples/garden_out.jpg"],
    ["examples/library_in.jpeg", "examples/library_out.jpg"],
]


gr.Interface(
    fn=inference,
    inputs=[gr.inputs.Image(type="pil")],
    outputs=gr.outputs.Image(type="pil"),
    title=title,
    description=description,
    article=article,
    examples=None,
    allow_flagging=False,
    allow_screenshot=False,
    enable_queue=True,
).launch()
