import numpy as np
import math
from skimage.metrics import structural_similarity


def accuracy_of_test(test_output, pred_output):
    batch_size = test_output.size(0)
    mse, psnr, ssim = 0, 0, 0

    test_output = test_output.detach().numpy()
    pred_output = pred_output.detach().numpy()

    for batch in range(batch_size):
        cur_test_output = test_output[batch].reshape(256, 256)
        cur_pred_output = pred_output[batch].reshape(256, 256)

        mse = np.mean((cur_test_output - cur_pred_output) ** 2)
        psnr += 20 * math.log10(1 / math.sqrt(mse))
        ssim += structural_similarity(cur_test_output[:180, :240], cur_pred_output[:180, :240])

    return ssim / batch_size, psnr / batch_size
