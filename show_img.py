#%%

import matplotlib.pyplot as plt
from PIL import Image

# Path to your image
image_path = 'datasets/scienceqa/images/test/85/image.png'

# Open the image
image = Image.open(image_path)

# Display the image
plt.imshow(image)
plt.axis('off')  # Turn off axis numbers
plt.show()

# %%
