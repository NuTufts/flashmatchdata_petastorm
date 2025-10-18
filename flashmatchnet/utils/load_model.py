import os,sys
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# Model
try:
    from flashmatchnet.model.lightmodel_siren import LightModelSiren
    from flashmatchnet.model.flashmatchMLP import FlashMatchMLP
except Exception as e:
    print("Trouble loading Siren Model")
    print("Did you active the siren-pytorch submodule? :: git submodule init; git submodule update")
    print("Did you set the environment varibles? :: source setenv_flashmatchdata.sh")
    print(e)
    sys.exit(1)

def load_model(config: Dict[str, Any], is_distributed: bool, local_rank: int) -> LightModelSiren:

    if 'model' in config:
        model_config = config['model']
    else:
        raise ValueError('config dict must have key "model"')

    network_type = model_config.get('network_type')
    device = torch.device(model_config.get('device','cpu'))

    if network_type=='mlp':
        # Create MLP for embeddings
        flashmlp_config = model_config['mlp']
        model = FlashMatchMLP(
            input_nfeatures=flashmlp_config['input_nfeatures'],
            hidden_layer_nfeatures=flashmlp_config['hidden_layer_nfeatures'],
            norm_layer=flashmlp_config['norm_layer']
        ).to(device)
    elif network_type=='lightmodelsiren':
    
        # Create SIREN network
        siren_config = model_config['lightmodelsiren']
    
        # Handle final activation
        if siren_config.get('final_activation') == 'identity':
            final_activation = nn.Identity()
        else:
            raise ValueError(f"Invalid final_activation: {siren_config.get('final_activation')}")
    
        # Create SIREN model
        model = LightModelSiren(
            dim_in=siren_config['dim_in'],
            dim_hidden=siren_config['dim_hidden'],
            dim_out=siren_config['dim_out'],
            num_layers=siren_config['num_layers'],
            w0_initial=siren_config['w0_initial'],
            final_activation=final_activation
        ).to(torch.device('cpu'))
    else:
        raise ValueError("Model network type not recognized: ",network_type)

    # Load checkpoint
    checkpoint_file = model_config.get('checkpoint',None)
    if checkpoint_file is not None:
        print("Loading Model from Checkpoint")
        print("  checkpoint file: ",checkpoint_file)
        if not os.path.exists( checkpoint_file ):
            checkpoint_file = os.environ['FLASHMATCH_BASEDIR']+"/"+checkpoint_file
        state_dict = torch.load( checkpoint_file, map_location=torch.device('cpu') )
        print("  stat_dict keys: ",state_dict.keys())
        print("  model_states keys: ",state_dict['model_states'].keys())
        model.load_state_dict( state_dict['model_states']['siren'] )

    model = model.to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=config['distributed'].get('gradient_as_bucket_view', True),
            find_unused_parameters=config['distributed'].get('find_unused_parameters', False),
            broadcast_buffers=config['distributed'].get('broadcast_buffers', True)
        )
    
    return model
