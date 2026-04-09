import os
import torch
import zipfile
import io
import numpy as np
from stable_baselines3 import SAC

print("--- STARTING WEIGHTS EXTRACTION ---")

input_zip = "sac_cayote_model_50000_steps.zip"
output_model = "sac_cayote_v1_compat"

if not os.path.exists(input_zip):
    print(f"Error: {input_zip} not found.")
    exit()

try:
    # 1. Manually extract the 'data' and 'policy.pth' from the zip
    # This avoids using the broken SAC.load()
    with zipfile.ZipFile(input_zip, 'r') as archive:
        # Load the PyTorch weights directly
        with archive.open('policy.pth') as weights_file:
            # We use map_location to ensure it loads on CPU/current device
            state_dict = torch.load(weights_file, map_location=torch.device('cpu'))
            print("Successfully extracted Neural Network weights.")

    # 2. To rebuild the model, we need the "Spaces" (Observation/Action size)
    # Since we can't load the model to ask it, we'll try to infer it from the weights.
    # For SAC, 'actor.latent_pi.0.weight' usually tells us the observation size.
    obs_size = state_dict['actor.latent_pi.0.weight'].shape[1]
    # 'actor.mu.weight' tells us the action size
    action_size = state_dict['actor.mu.weight'].shape[0]
    
    print(f"Detected Model Shapes: Obs={obs_size}, Actions={action_size}")

    # 3. Create a BRAND NEW model in your current NumPy 1.26 environment
    # We use 'MlpPolicy' as that is the SB3 default for SAC
    from gymnasium import spaces
    import gymnasium as gym

    # Create dummy environment to satisfy SB3 requirements
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1, high=1, shape=(action_size,), dtype=np.float32)
    
    new_model = SAC("MlpPolicy", DummyEnv(), verbose=1)
    print("Created new empty SAC model.")

    # 4. Inject the old weights into the new model
    new_model.policy.load_state_dict(state_dict)
    print("Weights successfully injected into the new model!")

    # 5. Save it! This will now be in 1.26 format.
    new_model.save(output_model)
    print("-" * 30)
    print(f"SUCCESS! Saved as {output_model}.zip")
    print("This file should now work perfectly on your Jetson.")
    print("-" * 30)

except Exception as e:
    print(f"Weight extraction failed: {e}")
    import traceback
    traceback.print_exc()
