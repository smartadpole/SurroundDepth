import torch
import os
from networks import ResnetEncoder, DepthDecoder  # 假设这些模型定义在 models 模块中
import argparse
from tools.file import MkdirSimple

def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to save the ONNX model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    return parser.parse_args()

# Corrected function to load the model
def load_model(model_path, device):
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = ResnetEncoder(num_layers=34, pretrained=False)  # Replace with your actual encoder model class
    # Load encoder state dict
    encoder_state_dict = torch.load(encoder_path, map_location=device)
    encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(encoder_state_dict, strict=False)


    depth_decoder = DepthDecoder(skip=True, num_ch_enc=[int(ch) for ch in encoder.num_ch_enc])  # Pass the required argument

    # Load depth decoder state dict
    depth_decoder_state_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder_state_dict = {k: v for k, v in depth_decoder_state_dict.items() if k in depth_decoder.state_dict()}
    depth_decoder.load_state_dict(depth_decoder_state_dict, strict=False)

    encoder.to(device)
    depth_decoder.to(device)

    encoder.eval()
    depth_decoder.eval()

    return encoder, depth_decoder


class CombinedModel(torch.nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.encoder = ResnetEncoder(num_layers=34, pretrained=False)
        self.decoder = DepthDecoder(skip=True, num_ch_enc=[int(ch) for ch in self.encoder.num_ch_enc])

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output[("disp", 0)]

    def load(self, model_path, device):
        self.encoder, self.decoder = load_model(model_path, device)


def export_to_onnx(model_path, onnx_path, device):
    """Export the model to ONNX format."""
    model = CombinedModel()
    model.load(model_path, device)

    # Create dummy input for the model
    dummy_input = torch.randn(6, 3, 384, 640).to(device)  # Adjust the size as needed

    # Export the depth decoder
    with torch.no_grad():
        onnx_file = os.path.join(onnx_path, "depth_decoder.onnx")
        # depth_decoder = torch.jit.script(depth_decoder)
        torch.onnx.export(model, dummy_input, onnx_file,
                          export_params=False,  # store the trained parameter weights inside the model file
                          opset_version=12,  # the ONNX version to export the model to
                          do_constant_folding=True)

    print(f"Models exported to {onnx_path}")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)
    MkdirSimple(args.onnx_path)
    export_to_onnx(args.model_path, args.onnx_path, device)
