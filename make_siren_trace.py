import os,sys
import torch
import torch.nn as nn
import yaml
from typing import Dict, Any, Optional

# Model
from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
try:
    from flashmatchnet.model.lightmodel_siren import LightModelSiren
except Exception as e:
    print("Trouble loading Siren Model")
    print("Did you activate the siren-pytorch submodule? :: git submodule init; git submodule update")
    print("Did you set the environment variables? :: source setenv_flashmatchdata.sh")
    print(e)
    sys.exit(1)

# Input embeddings
from flashmatchnet.utils.coord_and_embed_functions import prepare_mlp_input_embeddings, prepare_mlp_input_variables
from flashmatchnet.utils.pmtpos import getPMTPosByOpDet

"""
make_siren_trace.py

This script converts a trained SIREN model to TorchScript format for deployment in C++ LibTorch environments.

The script:
1. Loads model configuration from YAML
2. Creates and loads the SIREN model weights
3. Traces/scripts the model to TorchScript
4. Saves the TorchScript model for C++ deployment
"""

def create_pmtpos():
    """Create PMT positions tensor"""
    pmtpos = torch.zeros((32, 3))
    for i in range(32):
        opdetpos = getPMTPosByOpDet(i, use_v4_geom=True)
        for j in range(3):
            pmtpos[i, j] = opdetpos[j]
    # Change coordinate system to 'tensor' system
    # Main difference is y=0 is at bottom of TPC
    pmtpos[:, 1] -= -117.0
    # The PMT x-positions need adjustment
    pmtpos[:, 0] = -20.0
    return pmtpos

def load_model(config: Dict[str, Any]) -> (FlashMatchMLP, LightModelSiren):
    """Load the SIREN model from checkpoint"""

    model_config = config['model']
    device = torch.device(config.get('torchscript', {}).get('device', 'cpu'))

    # Create MLP for embeddings (if needed for preprocessing)
    flashmlp_config = model_config['flashmlp']
    mlp = FlashMatchMLP(
        input_nfeatures=flashmlp_config['input_nfeatures'],
        hidden_layer_nfeatures=flashmlp_config['hidden_layer_nfeatures']
    ).to(device)

    # Create SIREN network
    siren_config = model_config['lightmodelsiren']

    # Handle final activation
    if siren_config.get('final_activation') == 'identity':
        final_activation = nn.Identity()
    else:
        raise ValueError(f"Invalid final_activation: {siren_config.get('final_activation')}")

    # Create SIREN model
    siren = LightModelSiren(
        dim_in=siren_config['dim_in'],
        dim_hidden=siren_config['dim_hidden'],
        dim_out=siren_config['dim_out'],
        num_layers=siren_config['num_layers'],
        w0_initial=siren_config['w0_initial'],
        final_activation=final_activation,
        use_logpe=config.get('use_logpe', False)
    ).to(device)

    # Load checkpoint
    checkpoint_file = model_config.get('checkpoint', None)
    if checkpoint_file is not None:
        print(f"Loading checkpoint from: {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location=device)
        print("State dict keys:", state_dict.keys())
        if 'model_states' in state_dict:
            print("Model states keys:", state_dict['model_states'].keys())
            siren.load_state_dict(state_dict['model_states']['siren'])
        else:
            # Try loading directly
            siren.load_state_dict(state_dict)
        print("Model loaded successfully")
    else:
        print("Warning: No checkpoint file specified")

    return mlp, siren

def create_example_input(config: Dict[str, Any], pmtpos: torch.Tensor, device: torch.device):
    """Create example input tensors for model tracing"""

    model_config = config['model']
    torchscript_config = config.get('torchscript', {})

    # Get dimensions from config
    num_voxels = torchscript_config.get('example_num_voxels', 100)
    batch_size = torchscript_config.get('example_batch_size', 1)

    # Create example coordinate and charge tensors
    # Coordinates: random positions in detector volume
    coord = torch.randn(batch_size, num_voxels, 3, device=device)
    coord[:, :, 0] = coord[:, :, 0] * 128 + 128  # X: [0, 256]
    coord[:, :, 1] = coord[:, :, 1] * 58.5 + 58.5  # Y: [0, 117]
    coord[:, :, 2] = coord[:, :, 2] * 518 + 518    # Z: [0, 1036]

    # Charge features: random charge values
    q_feat = torch.rand(batch_size, num_voxels, 3, device=device) * 100

    # Prepare input features
    if model_config.get('use_cos_input_embedding_vectors'):
        vox_feat, q = prepare_mlp_input_embeddings(coord, q_feat, pmtpos, vox_len_cm=1.0)
    else:
        # Reshape from (batch, nvoxels, 3) to (batch*nvoxels, 3)
        vox_feat = prepare_mlp_input_variables(
            coord.reshape(-1, 3),
            q_feat.reshape(-1, 3),
            pmtpos,
            vox_len_cm=1.0
        )

    # Reshape for model input
    Nbv, Npmt, K = vox_feat.shape
    vox_feat_flat = vox_feat.reshape((Nbv * Npmt, K))
    q_flat = vox_feat_flat[:, -1:]
    vox_feat_flat = vox_feat_flat[:, :-1]

    return vox_feat_flat, q_flat

def trace_siren_model(siren: LightModelSiren, config: Dict[str, Any], pmtpos: torch.Tensor):
    """Trace the SIREN model to TorchScript"""

    device = torch.device(config.get('torchscript', {}).get('device', 'cpu'))
    siren = siren.to(device)
    siren.eval()

    # Create example inputs
    example_vox_feat, example_q = create_example_input(config, pmtpos, device)

    print(f"Example input shapes:")
    print(f"  vox_feat: {example_vox_feat.shape}")
    print(f"  q: {example_q.shape}")

    # Try to trace the model
    print("\nTracing SIREN model...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(siren, (example_vox_feat, example_q))
        print("Successfully traced model")

        # Test the traced model
        print("\nTesting traced model...")
        with torch.no_grad():
            original_output = siren(example_vox_feat, example_q)
            traced_output = traced_model(example_vox_feat, example_q)

        max_diff = torch.max(torch.abs(original_output - traced_output)).item()
        print(f"Maximum difference between original and traced outputs: {max_diff}")

        if max_diff > 1e-5:
            print("Warning: Large difference detected between original and traced models")

        return traced_model

    except Exception as e:
        print(f"Error during tracing: {e}")
        print("\nAttempting to script the model instead...")

        try:
            scripted_model = torch.jit.script(siren)
            print("Successfully scripted model")

            # Test the scripted model
            with torch.no_grad():
                original_output = siren(example_vox_feat, example_q)
                scripted_output = scripted_model(example_vox_feat, example_q)

            max_diff = torch.max(torch.abs(original_output - scripted_output)).item()
            print(f"Maximum difference between original and scripted outputs: {max_diff}")

            return scripted_model

        except Exception as e2:
            print(f"Error during scripting: {e2}")
            raise

def main(config_path):
    """Main function to convert SIREN model to TorchScript"""

    # Load YAML configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add torchscript section if not present
    if 'torchscript' not in config:
        config['torchscript'] = {}

    torchscript_config = config['torchscript']
    device = torch.device(torchscript_config.get('device', 'cpu'))

    print("="*80)
    print("SIREN Model to TorchScript Converter")
    print("="*80)

    # Load the model
    mlp, siren = load_model(config)
    siren.eval()

    print("\nLoaded SIREN Model:")
    print("="*80)
    print(siren)
    print("="*80)

    # Create PMT positions
    pmtpos = create_pmtpos().to(device)
    print(f"\nPMT positions shape: {pmtpos.shape}")

    # Trace or script the model
    torchscript_model = trace_siren_model(siren, config, pmtpos)

    # Save the TorchScript model
    output_path = torchscript_config.get('output_path', 'siren_model.pt')

    # Save with example inputs for reference
    example_vox_feat, example_q = create_example_input(config, pmtpos, device)

    print(f"\nSaving TorchScript model to: {output_path}")

    # Save the traced/scripted model
    torch.jit.save(torchscript_model, output_path)

    # Also save metadata for C++ deployment
    metadata = {
        'input_dim': config['model']['lightmodelsiren']['dim_in'],
        'hidden_dim': config['model']['lightmodelsiren']['dim_hidden'],
        'output_dim': config['model']['lightmodelsiren']['dim_out'],
        'num_layers': config['model']['lightmodelsiren']['num_layers'],
        'w0_initial': config['model']['lightmodelsiren']['w0_initial'],
        'use_logpe': config.get('use_logpe', False),
        'example_vox_feat_shape': list(example_vox_feat.shape),
        'example_q_shape': list(example_q.shape),
        'pmtpos_shape': list(pmtpos.shape),
        'device': str(device)
    }

    # Save metadata as JSON
    import json
    metadata_path = output_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # Verify the saved model can be loaded
    print("\nVerifying saved model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()

    with torch.no_grad():
        test_output = loaded_model(example_vox_feat, example_q)

    print(f"Test output shape: {test_output.shape}")
    print(f"Test output range: [{test_output.min().item():.4f}, {test_output.max().item():.4f}]")

    print("\n" + "="*80)
    print("TorchScript conversion completed successfully!")
    print(f"Model saved to: {output_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("\nTo use in C++:")
    print("  torch::jit::script::Module module = torch::jit::load(\"" + output_path + "\");")
    print("="*80)

        


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_siren_trace.py <config.yaml>")
        print("\nExample config should include:")
        print("  model:")
        print("    checkpoint: /path/to/checkpoint.pt")
        print("    lightmodelsiren:")
        print("      dim_in: 7")
        print("      dim_hidden: 512")
        print("      dim_out: 1")
        print("      num_layers: 5")
        print("      w0_initial: 30.0")
        print("      final_activation: identity")
        print("  torchscript:")
        print("    device: cpu  # or cuda")
        print("    output_path: siren_model.pt")
        print("    example_num_voxels: 100")
        print("    example_batch_size: 1")
        sys.exit(1)

    main(sys.argv[1])



