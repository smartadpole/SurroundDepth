import argparse
import os
import cv2
import torch
import numpy as np
from networks import ResnetEncoder, DepthDecoder
from utils import visualize_depth
from tools.file import WalkImage, MkdirSimple, FILE_SUFFIX

def load_model(model_path, device):
    """Load the model from the given path."""
    encoder = ResnetEncoder(num_layers=18, pretrained=False)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)

    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    depth_decoder.load_state_dict(torch.load(depth_decoder_path, map_location=device))

    encoder.to(device)
    depth_decoder.to(device)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder

def preprocess_image(image_path, input_width, input_height):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape
    image = cv2.resize(image, (input_width, input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0)
    return tensor, original_width, original_height

def predict_depth(encoder, depth_decoder, image):
    """Predict depth for a single image."""
    with torch.no_grad():
        features = encoder(image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        scaled_disp, depth = disp_to_depth(disp, min_depth=0.1, max_depth=100.0)
    return depth.squeeze().cpu().numpy()

def get_device(no_cuda = False):
    """Get the device to run the model on."""
    if torch.cuda.is_available() and not no_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_image_paths(image_path):
    """Get a list of image paths from a directory or a single image path."""
    if os.path.isfile(image_path):
        if image_path.endswith(FILE_SUFFIX):
            return [image_path]
        else:
            image_list = []
            with open(image_path, "r") as file:
                image_list = file.readlines()
                image_list = [f.strip() for f in image_list]
            return image_list
    elif os.path.isdir(image_path):
        return WalkImage(image_path)
    else:
        raise Exception("Cannot find image_path: {}".format(image_path))

def main():
    parser = argparse.ArgumentParser(description="Inference Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--image_paths", type=str, required=True, help="input image dir or image path or paths list file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the results.")
    parser.add_argument("--width", type=int, default=640, help="Input image width.")
    parser.add_argument("--height", type=int, default=384, help="Input image height.")
    args = parser.parse_args()


    device = get_device()
    image_paths = get_image_paths(args.image_paths)

    # Load model
    encoder, depth_decoder = load_model(args.model_path, device)

    # Process each image
    os.makedirs(args.output_dir, exist_ok=True)

    for image_path in image_paths:
        # Preprocess image
        tensor, original_width, original_height = preprocess_image(image_path, args.width, args.height)

        # Predict depth
        depth = predict_depth(encoder, depth_decoder, tensor)

        # Visualize depth
        depth_colored = visualize_depth(depth)

        # Save results
        base_name = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"depth_{base_name}")
        cv2.imwrite(output_path, depth_colored)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()