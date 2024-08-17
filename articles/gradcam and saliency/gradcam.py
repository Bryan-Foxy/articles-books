import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for visualizing the regions
    of an image that are important for a given classification decision made by a convolutional neural network.

    Attributes:
        model (torch.nn.Module): Pretrained model.
        image (PIL.Image): Input image.
        transform (torchvision.transforms): Transformations to be applied to the input image.
        target_layer (str): The target layer for Grad-CAM.
        activations (torch.Tensor): Activations of the target layer.
        gradients (torch.Tensor): Gradients of the target layer.
    """

    def __init__(self, image, model, target_layer='layer4.1.conv2', transform=None):
        """
        Initializes the GradCAM object with the image, model, and target layer.
        
        Args:
            image (PIL.Image): Input image.
            model (torch.nn.Module): Pretrained model.
            target_layer (str): The target layer for Grad-CAM. Default is 'layer4.1.conv2'.
            transform (torchvision.transforms): Transformations to be applied to the input image. Default is None.
        """
        super().__init__()
        self.model = model
        self.image = image
        self.transform = transform
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Hook the target layer
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks to the target layer to capture activations and gradients.
        """
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def forward(self):
        """
        Performs a forward pass through the model to get predictions.

        Returns:
            tuple: Predictions from the model and the transformed input image.
        """
        self.model.eval()
        if self.transform:
            input_img = self.transform(self.image).unsqueeze(0)
        else:
            input_img = transforms.ToTensor()(self.image).unsqueeze(0)
        
        input_img.requires_grad = True
        preds = self.model(input_img)
        return preds, input_img

    def generate_cam(self, class_idx):
        """
        Generates the class activation map (CAM) for the specified class index.

        Args:
            class_idx (int): The index of the class for which the CAM is generated.

        Returns:
            np.ndarray: The class activation map.
        """
        preds, input_img = self.forward()
        score = preds[0, class_idx]
        self.model.zero_grad()
        score.backward()

        gradients = self.gradients.data.numpy()[0]
        activations = self.activations.data.numpy()[0]
        
        # Compute the weights
        weights = np.mean(gradients, axis=(1, 2))

        # Compute the weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to the CAM
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        # Resize the CAM to match the input image size
        cam = cv2.resize(cam, (self.image.size[0], self.image.size[1]))
        return cam

    def visualize(self, class_idx):
        """
        Visualizes the Grad-CAM heatmap for the specified class index.

        Args:
            class_idx (int): The index of the class for which the heatmap is generated.
        """
        cam = self.generate_cam(class_idx)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        image = np.array(self.image) / 255
        cam_image = heatmap + np.float32(image)
        cam_image = cam_image / np.max(cam_image)

        plt.title('Gradient CAM')
        plt.imsave('cam_image.jpg', cam_image)
        plt.imshow(cam_image)
        plt.colorbar()
        plt.axis('off')
        plt.show()
