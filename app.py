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
    repo_id="Alpha-VLLM/Lumina-Next-T2I", local_dir="/home/user/app/checkpoints"
)

import argparse
import builtins
import json
import random
import socket

import spaces
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import gradio as gr
import numpy as np

import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

from PIL import Image
from safetensors.torch import load_file

import models
from transport import create_transport, Sampler

print(f"Is CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

description = """
    # Lumina Next Text-to-Image
    
    #### Lumina-Next-T2I is a 2B `Next-DiT` model with `Gemma-2B` text encoder.
    
    #### Demo current model: `Lumina-Next-T2I`

    #### Lumina-Next supports higher-order solvers. <span style='color: orange;'>It can generate images with merely 10 steps without any distillation.
 
"""
hf_token = os.environ["HF_TOKEN"]


class ModelFailure:
    pass


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(
    prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True
):

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


def load_models(args, master_port, rank):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    # Override the built-in print with the new version
    builtins.print = print

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.precision
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Creating lm: Gemma-2B")
    text_encoder = (
        AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=dtype,
            device_map=device,
            # device_map="cuda",
            token=hf_token,
        )
        .get_decoder()
        .eval()
    )
    cap_feat_dim = text_encoder.config.hidden_size
    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-2b",
        add_bos_token=True,
        add_eos_token=True,
        token=hf_token,
    )
    tokenizer.padding_side = "right"

    print(f"Creating vae: sdxl-vae")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-vae",
        torch_dtype=torch.float32,
    ).to(device)

    print(f"Creating DiT: Next-DiT")
    # latent_size = train_args.image_size // 8
    model = models.__dict__["NextDiT_2B_GQA_patch2"](
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    # model.eval().to("cuda", dtype=dtype)
    model.eval().to(device, dtype=dtype)

    assert train_args.model_parallel_size == args.num_gpus
    if args.ema:
        print("Loading ema model.")
    ckpt = load_file(
        os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
        ),
    )
    model.load_state_dict(ckpt, strict=True)

    return text_encoder, tokenizer, vae, model


@torch.no_grad()
def infer_ode(args, infer_args, text_encoder, tokenizer, vae, model):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.precision
    ]
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    # loading model to gpu
    # text_encoder = text_encoder.cuda()
    # vae = vae.cuda()
    # model = model.to("cuda", dtype=dtype)

    with torch.autocast("cuda", dtype):
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
            scaling_method=scaling_method,
            proportional_attn=proportional_attn,
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
            if args.likelihood:
                # assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"  # todo
                sample_fn = sampler.sample_ode_likelihood(
                    sampling_method=solver,
                    num_steps=num_sampling_steps,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            else:
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
                    cap_feats, cap_mask = encode_prompt(
                        [cap] + [neg_cap],
                        text_encoder,
                        tokenizer,
                        0.0,
                    )
                else:
                    cap_feats, cap_mask = encode_prompt(
                        [cap] + [""],
                        text_encoder,
                        tokenizer,
                        0.0,
                    )
            cap_mask = cap_mask.to(cap_feats.device)

            model_kwargs = dict(
                cap_feats=cap_feats,
                cap_mask=cap_mask,
                cfg_scale=cfg_scale,
            )

            if proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
            if do_extrapolation and scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size ** 2)
            else:
                model_kwargs["scale_factor"] = 1.0

            print(f"> scale factor: {model_kwargs['scale_factor']}")

            print("> start sample")
            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:1]

            factor = 0.18215 if train_args.vae != "sdxl" else 0.13025
            print(f"vae factor: {factor}")

            samples = vae.decode(samples / factor).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            img = to_pil_image(samples[0].float())

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
    group.add_argument(
        "--sample-eps", type=float, help="sampling in the transport model."
    )
    group.add_argument(
        "--train-eps", type=float, help="training to stabilize the learning process."
    )


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
    group.add_argument(
        "--reverse", action="store_true", help="run the ODE solver in reverse."
    )
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )


def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument(
        "--sampling-method",
        type=str,
        default="Euler",
        choices=["Euler", "Heun"],
        help="the numerical method used for sampling the stochastic differential equation: 'Euler' for simplicity or 'Heun' for improved accuracy.",
    )
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="sigma",
        choices=[
            "constant",
            "SBDM",
            "sigma",
            "linear",
            "decreasing",
            "increasing-decreasing",
        ],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument(
        "--diffusion-norm",
        type=float,
        default=1.0,
        help="Normalizes the diffusion coefficient, affecting the scale of the stochastic component.",
    )
    group.add_argument(
        "--last-step",
        type=none_or_str,
        default="Mean",
        choices=[None, "Mean", "Tweedie", "Euler"],
        help="form of last step taken in the SDE",
    )
    group.add_argument(
        "--last-step-size", type=float, default=0.04, help="size of the last step taken"
    )


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser()
    mode = "ODE"

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default="/home/user/app/checkpoints")
    parser.add_argument("--ema", type=bool, default=True)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])

    parse_transport_args(parser)
    parse_ode_args(parser)
    args = parser.parse_known_args()[0]

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    args.sampler_mode = mode

    text_encoder, tokenizer, vae, model = load_models(args, 60001, 0)

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
                    value="",
                    placeholder="Enter a negative caption.",
                )
                with gr.Row():
                    res_choices = ["1024x1024", "512x2048", "2048x512"] + [
                        "(Extrapolation) 2048x1920",
                        "(Extrapolation) 1920x2048",
                        "(Extrapolation) 1664x1664",
                        "(Extrapolation) 1536x2560",
                        "(Extrapolation) 2048x1024",
                        "(Extrapolation) 1024x2048",
                    ]
                    resolution = gr.Dropdown(
                        value=res_choices[0], choices=res_choices, label="Resolution"
                    )
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1,
                        maximum=70,
                        value=10,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=1,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                with gr.Accordion(
                    "Advanced Settings for Resolution Extrapolation", open=False
                ):
                    with gr.Row():
                        solver = gr.Dropdown(
                            value="midpoint",
                            choices=["euler", "midpoint", "rk4"],
                            label="solver",
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
                    with gr.Row():
                        scale_methods = gr.Dropdown(
                            value="Time-aware",
                            choices=["Time-aware", "None"],
                            label="Rope scaling method",
                        )
                        proportional_attn = gr.Checkbox(
                            value=True,
                            interactive=True,
                            label="Proportional attention",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    reset_btn = gr.ClearButton(
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
                            proportional_attn,
                        ],
                        value="Cancel",
                        variant="stop",
                        
                    )
            with gr.Column():
                output_img = gr.Image(
                    label="Lumina Generated image",
                    interactive=False,
                    format="png",
                    show_label=False
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            gr.Examples(
                [
                    ["👽🤖👹👻"],
                    ["🐔 playing 🏀"],
                    ["☃️ with 🌹 in the ❄️"],
                    ["🐶 wearing 😎  flying on 🌈 "],
                    ["A small 🍎 and 🍊 with 😁 emoji in the Sahara desert"],
                    ["Astronaut on Mars During sunset"],
                    [
                        "A scared cute rabbit in Happy Tree Friends style and punk vibe."
                    ],
                    ["A humanoid eagle soldier of the First World War."],  # noqa
                    [
                        "A cute Christmas mockup on an old wooden industrial desk table with Christmas decorations and bokeh lights in the background."
                    ],
                    [
                        "A front view of a romantic flower shop in France filled with various blooming flowers including lavenders and roses."
                    ],
                    [
                        "An old man, portrayed as a retro superhero, stands in the streets of New York City at night"
                    ],
                    [
                        "many trees are surrounded by a lake in autumn colors, in the style of nature-inspired imagery, havencore, brightly colored, dark white and dark orange, bright primary colors, environmental activism, forestpunk"
                    ],
                    [
                        "A fluffy mouse holding a watermelon, in a magical and colorful setting, illustrated in the style of Hayao Miyazaki anime by Studio Ghibli."
                    ],
                    ["孤舟蓑笠翁"],
                    ["两只黄鹂鸣翠柳"],
                    ["大漠孤烟直，长河落日圆"],
                    ["秋风起兮白云飞，草木黄落兮雁南归"],
                    ["味噌ラーメン, 最高品質の浮世絵、江戸時代。"],
                    ["東京タワー、最高品質の浮世絵、江戸時代。"],
                    ["도쿄 타워, 최고 품질의 우키요에, 에도 시대"],
                    [
                        "Tour de Tokyo, estampes ukiyo-e de la plus haute qualité, période Edo"
                    ],
                    ["Токийская башня, лучшие укиё-э, период Эдо"],
                    ["Tokio-Turm, hochwertigste Ukiyo-e, Edo-Zeit"],
                    [
                        "Inka warrior with a war make up, medium shot, natural light, Award winning wildlife photography, hyperrealistic, 8k resolution"
                    ],
                    [
                        "Character of lion in style of saiyan, mafia, gangsta, citylights background, Hyper detailed, hyper realistic, unreal engine ue5, cgi 3d, cinematic shot, 8k"
                    ],
                    [
                        "In the sky above, a giant, whimsical cloud shaped like the 😊 emoji casts a soft, golden light over the scene"
                    ],
                    [
                        "Cyberpunk eagle, neon ambiance, abstract black oil, gear mecha, detailed acrylic, grunge, intricate complexity, rendered in unreal engine 5, photorealistic, 8k"
                    ],
                    [
                        "close-up photo of a beautiful red rose breaking through a cube made of ice , splintered cracked ice surface, frosted colors, blood dripping from rose, melting ice, Valentine’s Day vibes, cinematic, sharp focus, intricate, cinematic, dramatic light"
                    ],
                    [
                        "3D cartoon Fox Head with Human Body, Wearing Iridescent Holographic Liquid Texture & Translucent Material Sun Protective Shirt, Boss Feel, Nike or Addidas Sun Protective Shirt, WitchPunk, Y2K Style, Green and blue, Blue, Metallic Feel, Strong Reflection, plain background, no background, pure single color background, Digital Fashion, Surreal Futurism, Supreme Kong NFT Artwork Style, disney style, headshot photography for portrait studio shoot, fashion editorial aesthetic, high resolution in the style of HAPE PRIME NFT, NFT 3D IP Feel, Bored Ape Yacht Club NFT project Feel, high detail, fine luster, 3D render, oc render, best quality, 8K, bright, front lighting, Face Shot, fine luster, ultra detailed"
                    ],
                ],
                [cap],
                label="Examples",
                examples_per_page=22,
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
                scale_methods,
                proportional_attn,
            ],
            [output_img, gr_metadata],
        )

    demo.queue().launch()


if __name__ == "__main__":
    main()
