import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def saliency(image, model, transform = None): 
    """
    Generates a saliency map for an image using a pretrained PyTorch model.

    The saliency map highlights the regions of the image that most significantly
    influence the model's prediction. It's a useful tool for understanding which
    parts of the image are important to the model.

    Args:
        image (PIL Image): The input image to analyze.
        model (torch.nn.Module): The pretrained PyTorch model.
        transform (callable, optional): An optional transform to apply to the image. 
                                        If not provided, a default ToTensor transform is used.

    Returns:
        tuple: A tuple containing the following:
            - idx (torch.Tensor): Index of the predicted class with the highest score.
            - score (torch.Tensor): Score (confidence) of the predicted class.

    Formula:
        The saliency map is calculated using the gradient backpropagation method:

        Saliency(x, y) = | ∂y_c / ∂I(x, y) |

        where:
            - Saliency(x, y) is the saliency value for pixel (x, y)
            - y_c is the score of the predicted class
            - I(x, y) is the value of pixel (x, y) in the input image
    """
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval() # Set to evaluate
    input = image.copy()
    if transform:
        input = transform(input)
    else:
        input = transforms.ToTensor()(input) 

    
    input.unsqueeze_(0)
    input.requires_grad = True
    preds = model(input)
    score, idx = torch.max(preds, 1)
    score.backward() 
    slc, _ = torch.max(torch.abs(input.grad[0]), dim = 0) 
    print(f"Indice of the best score: {idx}")
    print(f"Score: {score}")

    fig, ax = plt.subplots(1,2)
    fig.suptitle("Saliency map")
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].imshow(slc.numpy(), cmap = 'hot')
    ax[1].axis('off')

    fig.savefig('saliency.jpg')
    fig.show()