import sys
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


# Model
try:
    from flashmatchnet.model.lightmodel_siren import LightModelSiren
except Exception as e:
    print("Trouble loading Siren Model")
    print("Did you active the siren-pytorch submodule? :: git submodule init; git submodule update")
    print("Did you set the environment varibles? :: source setenv_flashmatchdata.sh")
    print(e)
    sys.exit(1)

def load_model(config: Dict[str, Any]) -> LightModelSiren:

    if 'model' in config:
        model_config = config['model']
    else:
        raise ValueError('config dict must have key "model"')

    device = torch.device(model_config.get('device','cpu'))
    
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
        final_activation=final_activation
    ).to(torch.device('cpu'))

    # Load checkpoint
    checkpoint_file = model_config.get('checkpoint',None)
    if checkpoint_file is not None:
        print("Loading Model from Checkpoint")
        print("  checkpoint file: ",checkpoint_file)
        state_dict = torch.load( checkpoint_file, map_location=torch.device('cpu') )
        print("  stat_dict keys: ",state_dict.keys())
        print("  model_states keys: ",state_dict['model_states'].keys())
        siren.load_state_dict( state_dict['model_states']['siren'] )

    siren = siren.to(device)
    
    return siren