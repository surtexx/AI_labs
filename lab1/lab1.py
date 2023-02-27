import numpy as np
import matplotlib.pyplot as plt
from skimage import io

images = np.array([np.load(f"images/car_{i}.npy") for i in range(9)])

images_sum = np.sum(images)

each_image_sum = [np.sum(images[i]) for i in range(9)]

index_max_image = np.argmax(each_image_sum)

imagine_medie = np.mean(images, axis=0)
io.imshow(imagine_medie.astype(np.uint8))
io.show()

deviatie = np.std(images)

normalized_images = [(each_image_sum[i] - imagine_medie) / deviatie for i in range(9)]
io.imshow(normalized_images[0].astype(np.uint8))
io.show()

cropped_images = images[:, 200:300, 280:400]
io.imshow(cropped_images[0].astype(np.uint8))
io.show()
