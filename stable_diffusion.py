import os
import time
import logging
import lpips
import torch
import argparse
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Load LPIPS model for perceptual similarity
lpips_model = lpips.LPIPS(net="alex").to("cuda:0")

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
    # Ensure tensors are properly formatted
    if not isinstance(img1, torch.Tensor):
        img1 = F.to_tensor(img1).unsqueeze(0).to("cuda:0")
    else:
        img1 = img1.unsqueeze(0).to("cuda:0")

    if not isinstance(img2, torch.Tensor):
        img2 = F.to_tensor(img2).unsqueeze(0).to("cuda:0")
    else:
        img2 = img2.unsqueeze(0).to("cuda:0")

    # Normalize images to [-1, 1] range
    img1 = img1 * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0

    return lpips_model(img1, img2).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing prompts", default="prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompts = load_prompts(args.prompt_file)

    # Load the baseline model
    logging.info("Loading Baseline Model...")
    baseline_pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")

    # Load the DeepCache model
    logging.info("Loading DeepCache Model...")
    deepcache_pipe = DeepCacheStableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")

    baseline_times = []
    deepcache_times_1 = []
    deepcache_times_2 = []
    lpips_scores_1 = []
    lpips_scores_2 = []

    for idx, prompt in enumerate(prompts):
        logging.info(f"Processing prompt {idx + 1}/{len(prompts)}: {prompt}")

        # Baseline Generation
        logging.info("Running Baseline...")
        start_time = time.time()
        ori_output = generate_image(baseline_pipe, prompt, seed)
        baseline_time = time.time() - start_time
        logging.info(f"Baseline Time: {baseline_time:.2f} seconds")
        baseline_times.append(baseline_time)

        # DeepCache Generation  - quad function
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

        # Compute LPIPS Score
        lpips_score = compute_lpips(ori_output[0], deepcache_output[0])
        logging.info(f"LPIPS Score: {lpips_score:.4f}")
        lpips_scores_1.append(lpips_score)
        
        # DeepCache Generation  - our function
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

        # Compute LPIPS Score
        lpips_score = compute_lpips(ori_output[0], deepcache_output[0])
        logging.info(f"LPIPS Score: {lpips_score:.4f}")
        lpips_scores_2.append(lpips_score)

        # Save Images
        # save_image([ori_output[0], deepcache_output[0]], f"output_{idx + 1}.png")
        # logging.info(f"Saved output_{idx + 1}.png")

    # Compute Averages
    avg_baseline_time = sum(baseline_times) / len(baseline_times)
    avg_deepcache_time_1 = sum(deepcache_times_1) / len(deepcache_times_1)
    avg_deepcache_time_2 = sum(deepcache_times_2) / len(deepcache_times_2)
    avg_lpips_1 = sum(lpips_scores_1) / len(lpips_scores_1)
    avg_lpips_2 = sum(lpips_scores_2) / len(lpips_scores_2)

    logging.info("======== Final Results ========")
    logging.info(f"Average Baseline Time: {avg_baseline_time:.2f} seconds")
    logging.info(f"Average DeepCache Time: {avg_deepcache_time_1:.2f} seconds")
    logging.info(f"Average LPIPS Score: {avg_lpips_1:.4f}")
    
    logging.info(f"Average DeepCache Time: {avg_deepcache_time_2:.2f} seconds")
    logging.info(f"Average LPIPS Score: {avg_lpips_2:.4f}")
