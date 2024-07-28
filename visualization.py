import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(image_path,transform):
    image = Image.open(image_path).convert('RGB')
    return image,transform(image).unsqueeze(0)

def visualize_prediction(original_image,probabilities,class_names):
    fig,axrr = plt.subplots(1,2,figsize=(14,7))

    #Display image
    axrr[0].imshow(original_image)
    axrr[0].axis('off')

    #Display predictions
    axrr[1].barh(class_names,probabilities)
    axrr[1].set_xlabel('Probability')
    axrr[1].set_title('Class Predictions')
    axrr[1].set_xlim(0,1)

    plt.tight_layout()
    plt.show()