import cv2 as cv
import numpy as np
from misc.utils import hsv2bgr


def random_hue():
    hue = None
    while (hue is None) or (35 < hue < 86) or (105 < hue < 135):
        hue = np.random.choice(range(1, 179), size=1)
    return hsv2bgr(hue)


def animate(img, skel, plants):
    all_p = img.copy()
    all_p[skel, ...] = [0, 0, 0]

    cv.namedWindow('image', cv.WINDOW_FULLSCREEN)
    h, w = skel.shape[:2]
    # colors = [(238, 255, 150), (0, 0, 255), (239, 0, 255)]
    for i, plant in enumerate(plants, 0):
        print(f'Plant: {i}')
        print(f'Num roots: {len(plant.roots)}')
        mask = np.zeros((h, w, 3))
        for root in plant.roots:
            print(root.edges)
            print(root._edges)
            print(root._split_node)
            print('\n\n\n')
            # color1 = colors[i]
            color1 = random_hue()
            points = np.array(root.points)
            for point in points:
                y, x = point

                mask[y, x, i] = 255
                all_p[y, x, :] = color1
                for y1 in range(y - 1, y + 2):
                    for x1 in range(x - 1, x + 2):
                        all_p[y1, x1, :] = color1
                cv.imshow('image', all_p.astype(np.uint8))
                cv.waitKey(1)
                # time.sleep(0.01)

        print('Stem:')
        points = np.array(plant.stem.points)
        print(plant.stem.edges)
        for point in points:
            y, x = point
            for y1 in range(y - 1, y + 2):
                for x1 in range(x - 1, x + 2):
                    all_p[y1, x1, :] = (0, 255, 0)
            cv.imshow('image', all_p.astype(np.uint8))
            cv.waitKey(1)
        # time.sleep(10)
    print('Done!')
    print('Press ESC to quit')

    cv.imshow('image', all_p.astype(np.uint8))
    while True:
        if cv.waitKey(20) & 0xFF == 27:
            break

    cv.destroyAllWindows()
