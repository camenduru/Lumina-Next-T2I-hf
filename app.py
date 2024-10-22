import os
import subprocess

subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)

os.makedirs("/home/user/app/checkpoints", exist_ok=True)
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Alpha-VLLM/Lumina-Next-SFT", local_dir="/home/user/app/checkpoints"
)

hf_token = os.environ["HF_TOKEN"]

import argparse
import builtins
import json
import math
import multiprocessing as mp
import os
import random
import socket
import traceback

from PIL import Image
import spaces
import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
from torchvision.transforms.functional import to_pil_image

import models
from transport import Sampler, create_transport


class ModelFailure:
    pass


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


@torch.no_grad()
def load_models(args, master_port, rank):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer

    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    # Override the built-in print with the new version
    builtins.print = print

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    print(f"Creating lm: Gemma-2B")
    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2b", torch_dtype=dtype, device_map=device, token=hf_token
    ).eval()
    cap_feat_dim = text_encoder.config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=hf_token, add_bos_token=True, add_eos_token=True)
    tokenizer.padding_side = "right"


    print(f"Creating vae: {train_args.vae}")
    vae = AutoencoderKL.from_pretrained(
        (f"stabilityai/sd-vae-ft-{train_args.vae}" if train_args.vae != "sdxl" else "stabilityai/sdxl-vae"),
        torch_dtype=torch.float32,
    ).cuda()

    print(f"Creating Next-DiT: {train_args.model}")
    # latent_size = train_args.image_size // 8
    model = models.__dict__[train_args.model](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to(device, dtype=dtype)

    if args.ema:
        print("Loading ema model.")
    ckpt = load_file(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
        )
    )
    model.load_state_dict(ckpt, strict=True)
    
    return text_encoder, tokenizer, vae, model

@torch.no_grad()
def infer_ode(args, infer_args, text_encoder, tokenizer, vae, model):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.precision
    ]
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    torch.cuda.set_device(0)
    
    with torch.autocast("cuda", dtype):
        while True:
            (
                cap,
                neg_cap,
                resolution,
                num_sampling_steps,
                cfg_scale,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ) = infer_args

            metadata = dict(
                cap=cap,
                neg_cap=neg_cap,
                resolution=resolution,
                num_sampling_steps=num_sampling_steps,
                cfg_scale=cfg_scale,
                solver=solver,
                t_shift=t_shift,
                seed=seed,
                # scaling_method=scaling_method,
                # scaling_watershed=scaling_watershed,
                # proportional_attn=proportional_attn,
            )
            print("> params:", json.dumps(metadata, indent=2))

            try:
                # begin sampler
                transport = create_transport(
                    args.path_type,
                    args.prediction,
                    args.loss_weight,
                    args.train_eps,
                    args.sample_eps,
                )
                sampler = Sampler(transport)
                sample_fn = sampler.sample_ode(
                    sampling_method=solver,
                    num_steps=num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                    reverse=args.reverse,
                    time_shifting_factor=t_shift,
                )
                # end sampler

                do_extrapolation = "Extrapolation" in resolution
                resolution = resolution.split(" ")[-1]
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                if int(seed) != 0:
                    torch.random.manual_seed(int(seed))
                z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1)

                with torch.no_grad():
                    if neg_cap != "":
                        cap_feats, cap_mask = encode_prompt([cap] + [neg_cap], text_encoder, tokenizer, 0.0)
                    else:
                        cap_feats, cap_mask = encode_prompt([cap] + [""], text_encoder, tokenizer, 0.0)

                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=cfg_scale,
                )
                if proportional_attn:
                    model_kwargs["proportional_attn"] = True
                    model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
                else:
                    model_kwargs["proportional_attn"] = False
                    model_kwargs["base_seqlen"] = None

                if do_extrapolation and scaling_method == "Time-aware":
                    model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size**2)
                    model_kwargs["scale_watershed"] = scaling_watershed
                else:
                    model_kwargs["scale_factor"] = 1.0
                    model_kwargs["scale_watershed"] = 1.0


                print("> start sample")
                samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]

                factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
                print(f"> vae factor: {factor}")
                samples = vae.decode(samples / factor).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                img = to_pil_image(samples[0].float())
                print("> generated image, done.")

                return img, metadata
            except Exception:
                print(traceback.format_exc())
                return ModelFailure()


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="/home/user/app/checkpoints")
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]
    args.sampler_mode = "ODE"

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    text_encoder, tokenizer, vae, model = load_models(args, 60001, 0)

    description = """
    # Lumina-Next-SFT

    Lumina-Next-SFT is a 2B Next-DiT model with Gemma-2B serving as the text encoder, enhanced through high-quality supervised fine-tuning (SFT).

    Demo current model: `Lumina-Next-SFT 1k Resolution`

    ### <span style='color: red;'> Lumina-Next-T2I enables zero-shot resolution extrapolation to 2k.

    ### Lumina-Next supports higher-order solvers ["euler", "midpoint"]. 
    ### <span style='color: orange;'>It can generate images with merely 10 steps without any distillation for 1K resolution generation.
    ### <span style='color: orange;'>Tip: For improved human portrait generation, please choose resolution at 1024x2048.

    ### To reduce waiting times, we are offering three parallel demos:
    
    Lumina-T2I 2B model: [[demo (supported 2k inference)](http://106.14.2.150:10020/)] [[demo (supported 2k inference)](http://106.14.2.150:10021/)] [[demo (supported 2k inference)](http://106.14.2.150:10022/)] [[demo (compositional generation)](http://106.14.2.150:10023/)]

    """
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(
                    lines=2,
                    label="Caption",
                    interactive=True,
                    value="Miss Mexico portrait of the most beautiful mexican woman, Exquisite detail, 30-megapixel, 4k, 85-mm-lens, sharp-focus, f:8, "
                    "ISO 100, shutter-speed 1:125, diffuse-back-lighting, award-winning photograph, small-catchlight, High-sharpness, facial-symmetry, 8k",
                    placeholder="Enter a caption.",
                )
                neg_cap = gr.Textbox(
                    lines=2,
                    label="Negative Caption",
                    interactive=True,
                    value="low resolution, low quality, blurry",
                    placeholder="Enter a negative caption.",
                )
                with gr.Row():
                    res_choices = [
                        "1024x1024",
                        "512x2048",
                        "2048x512",
                        "(Extrapolation) 1536x1536",
                        "(Extrapolation) 2048x1024",
                        "(Extrapolation) 1024x2048",
                        
                    ]
                    resolution = gr.Dropdown(value=res_choices[0], choices=res_choices, label="Resolution")
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1,
                        maximum=70,
                        value=30,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=25,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                with gr.Row():
                    solver = gr.Dropdown(
                        value="midpoint",
                        choices=["euler", "midpoint"],
                        label="Solver",
                    )
                    t_shift = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        interactive=True,
                        label="Time shift",
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=4.0,
                        interactive=True,
                        label="CFG scale",
                    )
                with gr.Accordion("Advanced Settings for Resolution Extrapolation", open=False, visible=False):
                    with gr.Row():
                        scaling_method = gr.Dropdown(
                            value="None",
                            choices=["None"],
                            label="RoPE scaling method",
                        )
                        scaling_watershed = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            interactive=True,
                            label="Linear/NTK watershed",
                            visible=False,
                        )
                    with gr.Row():
                        proportional_attn = gr.Checkbox(
                            value=True,
                            interactive=True,
                            label="Proportional attention",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_img = gr.Image(
                    label="Generated image",
                    interactive=False,
                    format="png"
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            gr.Examples(
                [
                    ["An old sailor, weathered by years at sea, stands at the helm of his ship, eyes scanning the horizon for signs of land, his face lined with tales of adventure and hardship."],  # noqa
                    ["A regal swan glides gracefully across the surface of a tranquil lake, its snowy white feathers ruffled by the gentle breeze."],  # noqa
                    ["A cunning fox, agilely weaving through the forest, its eyes sharp and alert, always ready for prey."],  # noqa
                    ["Inka warrior with a war make up, medium shot, natural light, Award winning wildlife photography, hyperrealistic, 8k resolution."],  # noqa
                    ["Quaint rustic witch's cabin by the lake, autumn forest background, orange and honey colors, beautiful composition, magical, warm glowing lighting, cloudy, dreamy masterpiece, Nikon D610, photorealism, highly artistic, highly detailed, ultra high resolution, sharp focus, Mysterious."],  # noqa
                ],
                [cap],
                label="Examples",
                examples_per_page=80,
            )

        @spaces.GPU(duration=200)
        def on_submit(*infer_args, progress=gr.Progress(track_tqdm=True),):
            result = infer_ode(args, infer_args, text_encoder, tokenizer, vae, model)
            if isinstance(result, ModelFailure):
                raise RuntimeError("Model failed to generate the image.")
            return result

        submit_btn.click(
            on_submit,
            [
                cap,
                neg_cap,
                resolution,
                num_sampling_steps,
                cfg_scale,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ],
            [output_img, gr_metadata],
        )

        def show_scaling_watershed(scaling_m):
            return gr.update(visible=scaling_m == "Time-aware")

        scaling_method.change(show_scaling_watershed, scaling_method, scaling_watershed)

    demo.queue().launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
