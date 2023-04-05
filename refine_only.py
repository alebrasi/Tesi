import cv2 as cv
import numpy as np
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from skimage.morphology import thin, medial_axis

from path_extraction.prune import prune3
from preprocess import locate_seed_line
from refine_mask import locate_seeds, refine_region_below, refine_region_above
from utils import find_file, show_image, DebugContext


def refine(img, mask, reference_img, invert_mask=True, binarize_mask=True, threshold_mask_value=1):
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    if invert_mask:
        mask = cv.bitwise_not(mask)

    if binarize_mask:
        _, mask = cv.threshold(mask, threshold_mask_value, 255, cv.THRESH_BINARY)

    h, w = img.shape[:2]

    seed_line_roi = [[148, 0], [300, w - 1]]

    M, left_pt, right_pt = locate_seed_line(img, seed_line_roi, 5)

    # Alignment in order to subdivide the above and below regions with even height
    # Must be done for image pyramids upscale
    if ((h - left_pt[1]) % 2) == 1:
        y = left_pt[1] + 1
        x1 = left_pt[0]
        x2 = right_pt[0]

        left_pt = (x1, y)
        right_pt = (x2, y)

    # Apply rotation matrix to image and mask
    img = cv.warpAffine(img, M, (w, h))
    mask = cv.warpAffine(mask, M, (w, h), cv.INTER_NEAREST)

    tmp = img.copy()
    cv.line(tmp, left_pt, right_pt, (0, 255, 0), 3)

    # ------------------- Mask preparation ------------------------------

    # Morphological closing for filling small gaps in the mask
    # ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    # mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, ker)

    matched_hist_img = match_histograms(img, reference_img, channel_axis=-1)
    segmented = cv.bitwise_and(matched_hist_img, matched_hist_img, mask=mask)

    region_below = segmented[left_pt[1]:, :]
    mask_above = mask[:left_pt[1], :]
    mask_below = mask[left_pt[1]:, :]
    # ------------ Seeds localization ----------------------------------
    top_left_pt, bottom_right_pt = seed_line_roi[0], seed_line_roi[1]
    br_y, br_x = bottom_right_pt
    seeds_roi = img[:br_y, :br_x]

    seeds_bb = locate_seeds(seeds_roi.copy(), dbg_ctx=None)
    # -----------------------------------------------------------------

    # -------------- Mask refinement ------------------
    h, w = segmented.shape[:2]
    refined_mask = np.zeros((h, w), dtype=np.uint8)

    refined_mask[left_pt[1]:, :] = refine_region_below(region_below, mask_below, dbg_ctx=None)
    refined_mask[:left_pt[1], :] = refine_region_above(mask_above)

    # Morphological closing inside seeds bounding box
    ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    for bb in seeds_bb:
        x, y, w, h = bb
        tmp = refined_mask[y:y + h, x:x + w]
        refined_mask[y:y + h, x:x + w] = cv.morphologyEx(tmp, cv.MORPH_CLOSE, ker)
    smoothed_mask = cv.medianBlur(refined_mask, 5)  # Edge smoothing

    # Filling gaps with area < 30 px
    cnts, _ = cv.findContours(smoothed_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(cnts):
        if cv.contourArea(cnt) < 30:
            cv.drawContours(smoothed_mask, cnts, i, 255, -1)

    inv_M = cv.invertAffineTransform(M)
    h, w = img.shape[:2]

    mask = cv.warpAffine(mask, inv_M, (w, h), cv.INTER_NEAREST_EXACT)
    smoothed_mask = cv.warpAffine(smoothed_mask, inv_M, (w, h), cv.INTER_NEAREST_EXACT)
    img = cv.warpAffine(img, inv_M, (w, h), cv.INTER_NEAREST_EXACT)

    return img, mask, smoothed_mask


def skeleton_and_prune(mask, branch_threshold=10):
    skeleton, dist = medial_axis(mask.astype(bool), return_distance=True, random_state=123)
    skeleton = skeleton.astype(np.uint8)
    cross_ker = np.array([[-1, 1, -1],
                          [1, -1, 1],
                          [-1, 1, -1]], dtype=np.uint8)
    cross_nodes = cv.morphologyEx(skeleton, cv.MORPH_HITMISS, cross_ker)
    skeleton = skeleton | cross_nodes

    pruned_skeleton = prune3(skeleton, branch_threshold)
    pruned_skeleton = thin(pruned_skeleton.astype(bool)).astype(np.uint8)

    return pruned_skeleton


def apply_watershed(img, mask, orig_mask):
    # kernel = np.ones((3, 3), np.uint8)
    # sure background area
    sure_bg = mask
    # sure_bg = cv.dilate(sure_bg, kernel, iterations=1)
    # Finding sure foreground area
    sure_fg = skeleton_and_prune(mask)
    # sure_fg = cv.dilate(sure_fg, kernel, iterations=0)
    sure_fg[sure_fg > 0] = 255

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    orig_mask[orig_mask > 0] = 255
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers[markers > 1] = 2
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv.watershed(img, markers)
    img1 = img.copy()
    img1[markers == -1] = [255, 0, 0]
    markers[markers == -1] = 2
    return img, markers


refined_mask1 = None
ix, iy = 0, 0
drawing = False
mode = True
brush_size = 3


def draw(event, x, y, flags, param):
    global refined_mask1, ix, iy, drawing, mode, brush_size
    ix, iy = 0, 0
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing is True:
            if mode:
                cv.circle(refined_mask1, (x, y), brush_size, (255, 255, 255), -1)
            else:
                cv.circle(refined_mask1, (x, y), brush_size, (0, 0, 0), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False


def refine_gui(img, refined_mask):
    global mode, refined_mask1, brush_size
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw)
    refined_mask1 = cv.cvtColor(refined_mask, cv.COLOR_GRAY2BGR)
    while True:
        blended = cv.addWeighted(img, 0.8, refined_mask1, 0.2, 0, 0)
        cv.imshow('image', blended)
        k = cv.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
            if mode:
                print('Draw tool')
            else:
                print('Erase tool')
        elif k == 27:
            break
        elif k == ord('+'):
            brush_size = brush_size + 1
            print(f'Brush size: {brush_size}')
        elif k == ord('-') and brush_size > 1:
            brush_size = brush_size - 1
            print(f'Brush size: {brush_size}')

    cv.destroyAllWindows()

    return cv.cvtColor(refined_mask1, cv.COLOR_BGR2GRAY)


if __name__ == '__main__':
    img_num = '1004R'
    reference_img_num = '995'

    img_path = find_file('/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/resize', f'{img_num}.jpg')
    mask_path = find_file('/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/segmentate_1024', f'{img_num}.png')
    reference_img_path = find_file('/home/alebrasi/Documents/tesi/Dataset/sessioni_crop/resize',
                                   f'{reference_img_num}.jpg')

    img = cv.imread(img_path)
    mask = cv.imread(mask_path)
    reference_img = cv.imread(reference_img_path)

    _, mask, refined_mask = refine(img, mask, reference_img)

    refined_mask2 = refine_gui(img, refined_mask)

    res, markers = apply_watershed(img, refined_mask2, mask)
    watersheded_mask = markers.astype(np.uint8)
    _, watersheded_mask = cv.threshold(watersheded_mask, 1, 255, cv.THRESH_BINARY)
    smoothed_mask = cv.medianBlur(watersheded_mask, 5)  # Edge smoothing

    diff = cv.subtract(refined_mask2, smoothed_mask)

    show_image([(img[..., ::-1], 'Immagine'), (mask, 'Maschera originale'),
                (refined_mask, 'Raffinata'), (refined_mask2, 'Annotazioni aggiuntive'),
                (smoothed_mask, 'Watershed con smoothing'), (diff, 'Differenza watershed-maschera annotata')], dbg_ctx=DebugContext(''), cols=3)
