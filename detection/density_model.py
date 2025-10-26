import os
import torch
import numpy as np
import cv2


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'csrnet_weights.pth')


class DensityModel:
    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu') -> None:
        """Load a pretrained density model if available.

        Notes:
        - This loader is intentionally generic. If a user saved only a state_dict,
          this file does not instantiate a network class; it's a simple convenience
          loader for setups where a full Module was saved with torch.save(module).
        """
        self.device = device
        self.model_path = model_path
        self.model = None

        if os.path.exists(model_path):
            try:
                # Try to load a full module first
                self.model = torch.load(model_path, map_location=device)
                # If it's a module, move to device and set eval()
                try:
                    self.model.to(device)
                    self.model.eval()
                except Exception:
                    # If it's not a module (e.g. state_dict), keep it as-is.
                    pass
            except Exception as e:
                # Keep model as None on failure; user can supply a proper model.
                print(f"Failed to load density model from {model_path}: {e}")

    def predict(self, frame: np.ndarray):
        """Predict a density map and estimated count for a BGR image.

        Input: BGR image (numpy array)
        Output: (density_map (H, W) | None, estimated_count (float))
        If no model loaded, returns (None, 0.0)
        """
        if self.model is None:
            return None, 0.0

        # Preprocess: convert to RGB, resize to model input, normalize
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = img.transpose(2, 0, 1).astype('float32') / 255.0
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            den = self.model(x)  # assume shape (1,1,H,W) or (1,H,W)
            den = den.squeeze().cpu().numpy()
            count = float(den.sum())

        # resize density map back to original frame size for visualization
        den_resized = cv2.resize(den, (frame.shape[1], frame.shape[0]))
        return den_resized, count


if __name__ == '__main__':
    dm = DensityModel()
    if dm.model is None:
        print('No density model found — place model at', MODEL_PATH)
    else:
        print('Density model loaded.')