import cv2 as cv
import numpy as np

def adjust_gamma(img, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype(np.uint8)

	return cv.LUT(img, table)

def f(img, alpha=2.0, beta=-200):
	alpha = np.array([alpha], dtype=np.uint16)
	
	img = np.clip((img * alpha) + beta, 0, 255).astype(np.uint8)
	return img

def remove_cc(img, area_thr, connectivity=8, algorithm=cv.CCL_DEFAULT):
    _, labels, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(img, connectivity, cv.CV_16U, algorithm)

    for label, stat in enumerate(stats):
        if stat[cv.CC_STAT_AREA] < area_thr:
            labels[labels == label] = 0

    labels[labels > 0] = 255

    return labels.astype(np.uint8)

def clahe_bgr(img, clip, grid_size):
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    h, l, s = cv.split(img)

    clahe = cv.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    l = clahe.apply(l)

    return cv.cvtColor(cv.merge([h, l, s]), cv.COLOR_HLS2BGR)

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    gray = gray[gray > 0]

    # Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = f(image, alpha=alpha, beta=beta)

    #auto_result = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)