import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

@torch.inference_mode()
def invert_single_step(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    target_t: float,
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    # order
    order: int = 2,
    # callback save function
    callback: Callable[[Tensor, dict], None] = None
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []

    for (current_t, prev_t) in zip(timesteps[:-1], timesteps[1:]):
        if prev_t == target_t:
            t_curr = current_t
            t_prev = prev_t
            info['t'] = t_prev if inverse else t_curr


    t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
    info['t'] = t_prev if inverse else t_curr

    # info['t'] = t_curr

    info['inverse'] = inverse
    info['second_order'] = False
    info['inject'] = inject_list[0]

    pred, info = model(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=t_vec,
        guidance=guidance_vec,
        info=info
    )

    if order == 2:
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
    
    # first order solver
    else:

        dt = t_prev - t_curr
        img = img + dt * pred

    log = {
        "t": info['t'],
        "latent": img.clone().cpu(),
        "score": pred.clone().cpu(),
    }

    if callback is not None:
        callback(log)

    return img, info

def denoise_single_step_to_x0(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    target_t: float,
    inverse,
    info, 
    guidance: float = 4.0,
    # order
    order: int = 2,
    # callback save function
    callback: Callable[[Tensor, dict], None] = None
):
    # this is ignored for schnell
    inject_list = [False]

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    t_vec = torch.full((img.shape[0],), target_t, dtype=img.dtype, device=img.device)
    info['t'] = target_t
    info['inverse'] = inverse
    info['second_order'] = False
    info['inject'] = inject_list[0]

    pred, info = model(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=vec,
        timesteps=t_vec,
        guidance=guidance_vec,
        info=info
    )

    img = img -target_t * pred

    log = {
        "t": info['t'],
        "latent": img.clone().cpu(),
        "score": pred.clone().cpu(),
    }

    if callback is not None:
        callback(log)

    return img, info

def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    # order
    order: int = 2,
    # callback save function
    callback: Callable[[Tensor, dict], None] = None,
    fastforward_steps: float = None
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []

    if fastforward_steps is not None:
        # Fastforward img
        random_gaussian = torch.randn_like(img)
        img = (1 - fastforward_steps) * img + fastforward_steps * random_gaussian
        timesteps = [t for t in timesteps if t >= fastforward_steps]

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr

        # info['t'] = t_curr

        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        if order == 2:
            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
            info['second_order'] = True
            pred_mid, info = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                info=info
            )

            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
            img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
        
        # first order solver
        else:
            img = img + (t_prev - t_curr) * pred

        log = {
            "t": info['t'],
            "latent": img.clone().cpu(),
            "score": pred.clone().cpu(),
        }

        if callback is not None:
            callback(log)

    return img, info

def denoise_starting_particular_step(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    target_t: float,
    inverse,
    info, 
    guidance: float = 4.0,
    # order
    order: int = 2,
    # callback save function
    callback: Callable[[Tensor, dict], None] = None
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    timesteps_below_target = [t for t in timesteps if t <= target_t]

    if inverse:
        timesteps_below_target = timesteps_below_target[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []

    for i, (t_curr, t_prev) in enumerate(zip(timesteps_below_target[:-1], timesteps_below_target[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        if order == 2:
            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
            info['second_order'] = True
            pred_mid, info = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                info=info
            )

            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
            img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
        
        # first order solver
        else:
            img = img + (t_prev - t_curr) * pred

        log = {
            "t": info['t'],
            "latent": img.clone().cpu(),
            "score": pred.clone().cpu(),
        }

        if callback is not None:
            callback(log)

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
