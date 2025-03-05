import os
import time
import logging
import lpips
import torch
import argparse
import clip
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Load LPIPS model for perceptual similarity
lpips_model = lpips.LPIPS(net="alex").to("cuda:0")

# Load CLIP Model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Set random seed
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load prompts from a .txt file
def load_prompts(file_path):
    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

# Generate an image using a given model
def generate_image(pipe, prompt, seed):
    set_random_seed(seed)
    return pipe(prompt, output_type="pt").images

# Compute LPIPS score
def compute_lpips(img1, img2):
    img1 = img1.unsqueeze(0).to("cuda:0")
    img2 = img2.unsqueeze(0).to("cuda:0")

    # Normalize images to [-1, 1] range
    img1 = img1 * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0

    return lpips_model(img1, img2).item()

# Compute CLIP Score
def compute_clip_score(image, prompt):
    image = transforms.ToPILImage()(image.cpu())  # Convert tensor to PIL
    image = clip_preprocess(image).unsqueeze(0).to(device)

    text = clip.tokenize([prompt]).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return (image_features @ text_features.T).item()

# Compute MS-SSIM Score
def compute_msssim(img1, img2):
    img1 = img1.unsqueeze(0).to("cuda:0")  # Add batch dimension and move to GPU
    img2 = img2.unsqueeze(0).to("cuda:0")

    return ms_ssim(img1, img2, data_range=1.0).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompts = load_prompts(args.prompt_file)

    # Load models
    logging.info("Loading Baseline Model...")
    baseline_pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")

    logging.info("Loading DeepCache Model...")
    deepcache_pipe = DeepCacheStableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")

    # Metrics storage
    baseline_times = []
    deepcache_times_1 = []
    deepcache_times_2 = []
    lpips_scores_1 = []
    lpips_scores_2 = []
    clip_scores_1 = []
    clip_scores_2 = []
    msssim_scores_1 = []
    msssim_scores_2 = []

    for idx, prompt in enumerate(prompts):
        logging.info(f"Processing prompt {idx + 1}/{len(prompts)}: {prompt}")

        # Baseline Generation
        logging.info("Running Baseline...")
        start_time = time.time()
        ori_output = generate_image(baseline_pipe, prompt, seed)
        baseline_time = time.time() - start_time
        logging.info(f"Baseline Time: {baseline_time:.2f} seconds")
        baseline_times.append(baseline_time)

        # DeepCache Generation - quad function
        logging.info("Running DeepCache for quad function...")
        start_time = time.time()
        deepcache_output = deepcache_pipe(
            prompt,
            cache_interval=5, cache_layer_id=0, cache_block_id=0,
            uniform=False, pow=1.4, center=15,
            output_type="pt", return_dict=True, function_type=1
        ).images
        deepcache_time_1 = time.time() - start_time
        logging.info(f"DeepCache Time: {deepcache_time_1:.2f} seconds")
        deepcache_times_1.append(deepcache_time_1)

        # Compute Metrics
        lpips_score = compute_lpips(ori_output[0], deepcache_output[0])
        clip_score = compute_clip_score(deepcache_output[0], prompt)
        msssim_score = compute_msssim(ori_output[0], deepcache_output[0])

        logging.info(f"LPIPS Score: {lpips_score:.4f}")
        logging.info(f"CLIP Score: {clip_score:.4f}")
        logging.info(f"MS-SSIM Score: {msssim_score:.4f}")

        lpips_scores_1.append(lpips_score)
        clip_scores_1.append(clip_score)
        msssim_scores_1.append(msssim_score)

        # DeepCache Generation - our function
        logging.info("Running DeepCache for our function...")
        start_time = time.time()
        deepcache_output = deepcache_pipe(
            prompt,
            cache_interval=5, cache_layer_id=0, cache_block_id=0,
            uniform=False, pow=1.4, center=15,
            output_type="pt", return_dict=True, function_type=2
        ).images
        deepcache_time_2 = time.time() - start_time
        logging.info(f"DeepCache Time: {deepcache_time_2:.2f} seconds")
        deepcache_times_2.append(deepcache_time_2)

        # Compute Metrics
        lpips_score = compute_lpips(ori_output[0], deepcache_output[0])
        clip_score = compute_clip_score(deepcache_output[0], prompt)
        msssim_score = compute_msssim(ori_output[0], deepcache_output[0])

        logging.info(f"LPIPS Score: {lpips_score:.4f}")
        logging.info(f"CLIP Score: {clip_score:.4f}")
        logging.info(f"MS-SSIM Score: {msssim_score:.4f}")

        lpips_scores_2.append(lpips_score)
        clip_scores_2.append(clip_score)
        msssim_scores_2.append(msssim_score)

    # Compute Averages
    def avg(lst):
        return sum(lst) / len(lst)

    logging.info("======== Final Results ========")
    logging.info(f"Average Baseline Time: {avg(baseline_times):.2f} seconds")
    logging.info(f"Average DeepCache Time - 1: {avg(deepcache_times_1):.2f} seconds")
    logging.info(f"Average LPIPS Score - 1: {avg(lpips_scores_1):.4f}")
    logging.info(f"Average CLIP Score - 1: {avg(clip_scores_1):.4f}")
    logging.info(f"Average MS-SSIM Score - 1: {avg(msssim_scores_1):.4f}")

    logging.info(f"Average DeepCache Time - 2: {avg(deepcache_times_2):.2f} seconds")
    logging.info(f"Average LPIPS Score - 2: {avg(lpips_scores_2):.4f}")
    logging.info(f"Average CLIP Score - 2: {avg(clip_scores_2):.4f}")
    logging.info(f"Average MS-SSIM Score - 2: {avg(msssim_scores_2):.4f}")
