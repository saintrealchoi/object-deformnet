import os
import cv2
import PIL.Image as pil
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


path ='/data/PhoCAL_release/sequence_1/depth/000183.png'
img = pil.open(path)
img_np = np.array(img)
cmaps = [plt.cm.gray, plt.cm.rainbow, plt.cm.Reds_r, plt.cm.Blues_r, plt.cm.cividis, plt.cm.cubehelix, plt.cm.magma]
# f, axes = plt.subplots(1, len(cmaps), figsize=(15, 6))
# for i in range(0, len(cmaps)):
#     axes[i].imshow(img_np, cmap=cmaps[i])
#     axes[i].set_xticks([]), axes[i].set_yticks([]), axes[i].set_title(cmaps[i].name)
# plt.savefig('../../assets/images/markdown_img/180630_1824_cat_gray_to_colormap.svg')
plt.imshow(img_np,cmap=cmaps[-1])
plt.show()