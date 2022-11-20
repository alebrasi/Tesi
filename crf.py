import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np
import cv2 as cv

# TODO: Togliere la roba che non serve
def CRF(mask, img, inference_steps):
    h, w = mask.shape[0], mask.shape[1]

    mask_crf = np.zeros((h, w, 3))
    mask_crf[..., 1] = mask
    mask_crf[..., 2] = cv.bitwise_not(mask)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    mask = mask_crf.astype(np.uint32)
    anno_lbl = mask[:,:,0] + (mask[:,:,1] << 8) + (mask[:,:,2] << 16)


    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    #print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    # Example using the DenseCRF2D code
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(inference_steps)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    #imwrite(fn_output, MAP.reshape(img.shape))

    return MAP.reshape(img.shape)