import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

import os
from collections import OrderedDict
import torch.utils.mobile_optimizer as mobile_optimizer

# ----------------------------
# Model Definitions (Modified for Quantization)
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False) # Bias often False for fusion with BN, but keep if trained with bias and no BN
        self.relu = nn.ReLU(inplace=False) # IMPORTANT: inplace=False for quantization
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.skip_add = nn.quantized.FloatFunctional() # For adding residual in FP32 domain if needed

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        # For static quantization, additions usually happen between dequantized tensors or require FloatFunctional
        return self.skip_add.add(out, residual)

    def fuse_model(self):
        # Fuse (Conv2d, ReLU)
        torch.quantization.fuse_modules(self, [['conv1', 'relu']], inplace=True)

class SRModel(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, features=64, num_res_blocks=5, upscale_factor=4):
        super(SRModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=False) # Added for potential fusion after conv1
        self.res_blocks = nn.Sequential(*[ResidualBlock(features) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(features, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x) # Apply ReLU
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'relu1']], inplace=True)
        for block in self.res_blocks:
            if hasattr(block, 'fuse_model'):
                block.fuse_model()

class RefinementNet(nn.Module):
    def __init__(self, channels=3, features=64, num_res_blocks=3):
        super(RefinementNet, self).__init__()
        self.conv_in = nn.Conv2d(channels, features, kernel_size=3, padding=1)
        self.relu_in = nn.ReLU(inplace=False) # Changed from F.relu for easier module access
        self.res_blocks = nn.Sequential(*[ResidualBlock(features) for _ in range(num_res_blocks)])
        self.conv_out = nn.Conv2d(features, channels, kernel_size=3, padding=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x # Store original input for the final addition
        out = self.conv_in(x)
        out = self.relu_in(out)
        out = self.res_blocks(out)
        out = self.conv_out(out)
        return self.skip_add.add(out, identity) # Add to original input

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv_in', 'relu_in']], inplace=True)
        for block in self.res_blocks:
            if hasattr(block, 'fuse_model'):
                block.fuse_model()

class TwoStageSRModel(nn.Module):
    def __init__(self, base_model, refinement_model):
        super(TwoStageSRModel, self).__init__()
        self.quant = torch.quantization.QuantStub()   # Marks start of quantized region
        self.base_model = base_model
        self.refinement_model = refinement_model
        self.dequant = torch.quantization.DeQuantStub() # Marks end of quantized region

    def forward(self, x):
        x = self.quant(x) # Quantize input
        base_output = self.base_model(x)
        refined_output = self.refinement_model(base_output)
        refined_output = self.dequant(refined_output) # Dequantize output
        return refined_output

    def fuse_model(self):
        # Fuse models within sub-modules
        if hasattr(self.base_model, 'fuse_model'):
            self.base_model.fuse_model()
        if hasattr(self.refinement_model, 'fuse_model'):
            self.refinement_model.fuse_model()
        # No top-level fusions here unless base_model output directly feeds into a specific layer of refinement_model
        # that can be fused, which is not the case here.

# ----------------------------
# Configuration
# ----------------------------
IN_CHANNELS = 9
OUT_CHANNELS = 3
FEATURES = 64
BASE_NUM_RES_BLOCKS = 5
UPSCALE_FACTOR = 4
REFINEMENT_NUM_RES_BLOCKS = 3

PTH_MODEL_PATH = "best.pth" # IMPORTANT: Update this path
PTL_OUTPUT_PATH_FLOAT = "lmf_cnn_mobile_float.ptl" # Original float model
PTL_OUTPUT_PATH_QUANTIZED = "lmf_cnn_mobile_quantized.ptl" # Quantized model

EXAMPLE_INPUT_SHAPE = (1, IN_CHANNELS, 120, 214) # (B, C, H, W)
NUM_CALIBRATION_BATCHES = 10 # Number of batches for calibration. More is better, up to a point.

# --- Calibration Data Loader (Placeholder - replace with your actual data) ---
# For best results, use a representative dataset.
# This is a placeholder using random data.
def get_calibration_data_loader(num_batches, batch_size, input_shape):
    # input_shape is (C, H, W)
    data = [torch.randn(batch_size, *input_shape[1:]) for _ in range(num_batches)]
    return data

def main():
    print(f"PyTorch version: {torch.__version__}")
    if not os.path.exists(PTH_MODEL_PATH):
        print(f"ERROR: Model checkpoint not found at '{PTH_MODEL_PATH}'")
        return

    # 1. Instantiate the model
    print("Instantiating model...")
    base_model = SRModel(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=FEATURES,
        num_res_blocks=BASE_NUM_RES_BLOCKS, upscale_factor=UPSCALE_FACTOR
    )
    refinement_net = RefinementNet(
        channels=OUT_CHANNELS, features=FEATURES, num_res_blocks=REFINEMENT_NUM_RES_BLOCKS
    )
    model = TwoStageSRModel(base_model, refinement_net)

    # 2. Load weights
    print(f"Loading weights from: {PTH_MODEL_PATH}")
    checkpoint = torch.load(PTH_MODEL_PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict, strict=False) # strict=False can be helpful if new ReLUs were added
    print("Weights loaded.")

    # ----------------------------------------------------------------------
    # ---- Static Quantization Steps ----
    # ----------------------------------------------------------------------
    quantized_model = model # Start with the original model structure

    # 3. Set model to evaluation mode (CRITICAL for quantization)
    quantized_model.eval()
    print("Model set to evaluation mode.")

    # 4. Fuse modules (Conv-ReLU, Conv-BN-ReLU, etc.)
    # This must be done *before* quantization preparation.
    print("Fusing model modules...")
    quantized_model.fuse_model() # Calls fuse_model() recursively if defined in submodules
    print("Model modules fused.")

    # 5. Specify quantization configuration
    # 'qnnpack' for ARM (mobile), 'fbgemm' for x86.
    # For mobile, 'qnnpack' is generally preferred.
    # PyTorch 1.7+ has improved mobile quantization. For older versions, 'fbgemm' might be more stable.
    quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    print(f"Using qconfig: {quantized_model.qconfig}")
    # You can also try per-channel quantization for conv layers for potentially better accuracy:
    # quantized_model.qconfig = torch.quantization.get_default_per_channel_qconfig('qnnpack')


    # 6. Prepare the model for static quantization. This inserts observers.
    print("Preparing model for static quantization...")
    torch.quantization.prepare(quantized_model, inplace=True)
    print("Model prepared.")

    # 7. Calibrate the model with representative data
    # The quality of calibration data is crucial for accuracy.
    print(f"Calibrating model with {NUM_CALIBRATION_BATCHES} batches...")
    # Use a small, representative dataset for calibration.
    # Here, we use random data as a placeholder.
    # The example input shape already includes batch size 1.
    calibration_data = get_calibration_data_loader(NUM_CALIBRATION_BATCHES, EXAMPLE_INPUT_SHAPE[0], EXAMPLE_INPUT_SHAPE)
    with torch.no_grad():
        for calib_input_batch in calibration_data:
            quantized_model(calib_input_batch) # Run data through the model
    print("Calibration complete.")

    # 8. Convert the model to a quantized version
    print("Converting model to quantized version...")
    # `convert` replaces observed modules with their quantized counterparts
    torch.quantization.convert(quantized_model, inplace=True)
    print("Model converted to quantized version.")

    # ----------------------------------------------------------------------
    # ---- Save Quantized Model to PyTorch Lite ----
    # ----------------------------------------------------------------------

    # 9. Create an example input tensor for tracing the *quantized* model
    example_input = torch.randn(EXAMPLE_INPUT_SHAPE, dtype=torch.float32)
    print(f"Using example input of shape: {example_input.shape} for tracing quantized model.")

    # 10. Trace the quantized model
    print("Tracing quantized model with torch.jit.trace...")
    try:
        # The quantized model still accepts float input, QuantStub handles the conversion
        traced_quantized_module = torch.jit.trace(quantized_model, example_input)
        print("Quantized model traced successfully.")
    except Exception as e:
        print(f"Error during JIT tracing of quantized model: {e}")
        return

    # 11. Optimize the traced quantized model for mobile
    print("Optimizing traced quantized model for mobile...")
    try:
        optimized_quantized_lite_module = mobile_optimizer.optimize_for_mobile(traced_quantized_module)
        print("Quantized model optimized for mobile successfully.")
    except Exception as e:
        print(f"Error during mobile optimization of quantized model: {e}")
        return

    # 12. Save the Lite Interpreter model (.ptl)
    print(f"Saving quantized Lite Interpreter model to: {PTL_OUTPUT_PATH_QUANTIZED}")
    try:
        optimized_quantized_lite_module._save_for_lite_interpreter(PTL_OUTPUT_PATH_QUANTIZED)
        print(f"Quantized Lite model saved successfully to {PTL_OUTPUT_PATH_QUANTIZED}")
        print(f"File size: {os.path.getsize(PTL_OUTPUT_PATH_QUANTIZED) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error saving quantized Lite Interpreter model: {e}")
        return

    print("\n--- Quantization and Conversion Complete ---")
    print(f"The quantized mobile-ready model is at: {PTL_OUTPUT_PATH_QUANTIZED}")

    # For comparison, you can also save the original float model in .ptl format
    # Re-instance and load weights for the float model if you modified the original `model` in-place
    print("\n--- Converting Original Float Model for Comparison (Optional) ---")
    base_model_float = SRModel(
        in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, features=FEATURES,
        num_res_blocks=BASE_NUM_RES_BLOCKS, upscale_factor=UPSCALE_FACTOR
    )
    refinement_net_float = RefinementNet(
        channels=OUT_CHANNELS, features=FEATURES, num_res_blocks=REFINEMENT_NUM_RES_BLOCKS
    )
    # Note: We need a TwoStageSRModel WITHOUT QuantStub/DeQuantStub for pure float export
    class TwoStageSRModelFloat(nn.Module): # Simpler version for float export
        def __init__(self, base_model, refinement_model):
            super(TwoStageSRModelFloat, self).__init__()
            self.base_model = base_model
            self.refinement_model = refinement_model
        def forward(self, x):
            base_output = self.base_model(x)
            refined_output = self.refinement_model(base_output)
            return refined_output

    model_float = TwoStageSRModelFloat(base_model_float, refinement_net_float)
    model_float.load_state_dict(new_state_dict, strict=False) # Use the same loaded weights
    model_float.eval()
    try:
        traced_float_module = torch.jit.trace(model_float, example_input)
        optimized_float_lite_module = mobile_optimizer.optimize_for_mobile(traced_float_module)
        optimized_float_lite_module._save_for_lite_interpreter(PTL_OUTPUT_PATH_FLOAT)
        print(f"Original Float Lite model saved to {PTL_OUTPUT_PATH_FLOAT}")
        print(f"File size: {os.path.getsize(PTL_OUTPUT_PATH_FLOAT) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error saving float Lite model: {e}")

if __name__ == '__main__':
    main()
