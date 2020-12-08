import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import read1line


def aps_image_processing(root):
    # APS images as labels
    images_lines = []

    with open(os.path.join(root, 'images.txt'), 'r') as images:
        for line in images:
            line = line.split()
            images_lines.append(line)

    for line in range(len(images_lines)):  # absolute dir
        images_lines[line][1] = root + images_lines[line][1]

    post_images_array = np.zeros((int(len(images_lines)), 256, 256), dtype='float16')

    # convert labels to grayscale images
    print('==> Get APS images..', '(', root, ')')
    for i in tqdm(range(0, len(images_lines)), ascii=True):
        im = Image.open(images_lines[i][1]).convert('L')
        data = np.asarray(im) / 255.0
        data = np.pad(data, ((0, 256 - data.shape[0]), (0, 256 - data.shape[1])), 'constant')
        post_images_array[i] = data

    root += 'events.txt'
    f = open(root, 'r')

    # get offset of each event.txt in order to avoid memory error
    lines = read1line.line2offset(f)
    lines_length = len(lines)
    print('events length:', len(lines))

    j = 0
    line = read1line.seek1line(f, lines, j)
    image_label_idx = [0]

    # get event line index of each APS image so we can separate them into different channels easily
    print('==> Get corresponding event line index of APS images..')
    for i in tqdm(range(0, len(images_lines)), ascii=True):
        while j < lines_length - 1 and float(line[0]) <= float(images_lines[i][0]):
            j += 1
            line = read1line.seek1line(f, lines, j)
        image_label_idx.append(j)

    f.close()
    return images_lines, image_label_idx, post_images_array, lines


def peseudo_image_processing(root, images_lines, image_label_idx, lines, channels, fixed=False):
    root += 'events.txt'
    f = open(root, 'r')

    """
    Event stack. Here is the example of counting the number of events 
    and limit the value in the range [-1, 1]. In 2-channel condition, we separate
    the positive and negative events into two channels, which is different
    from other channels.
    """

    print('==> Split events into channels..')
    if channels == 1:
        pre_images_array = np.zeros((int(len(images_lines)), 256, 256), dtype='float16')
        for i in tqdm(range(0, len(images_lines)), ascii=True):
            for j in range(image_label_idx[i], image_label_idx[i + 1]):
                line = read1line.seek1line(f, lines, j)
                if not fixed:
                    pre_images_array[i][int(line[2])][int(line[1])] = \
                        min(pre_images_array[i][int(line[2])][int(line[1])] + 0.1, 1) if int(line[3]) == 1 \
                        else max(pre_images_array[i][int(line[2])][int(line[1])] - 0.1, -1)
                else:
                    pre_images_array[i][int(line[2])][int(line[1])] = 0.5

    elif channels == 2:
        pre_images_array = np.zeros((int(len(images_lines)), 256, 256, 2), dtype='float16')
        for i in tqdm(range(0, len(images_lines)), ascii=True):
            for j in range(image_label_idx[i], image_label_idx[i + 1]):
                line = read1line.seek1line(f, lines, j)
                if not fixed:
                    if int(line[3]) == 1:
                        pre_images_array[i][int(line[2])][int(line[1])][0] = \
                            min(pre_images_array[i][int(line[2])][int(line[1])][0] + 0.1, 1)
                    else:
                        pre_images_array[i][int(line[2])][int(line[1])][1] = \
                            min(pre_images_array[i][int(line[2])][int(line[1])][1] + 0.1, 1)
                else:
                    if int(line[3]) == 1:
                        pre_images_array[i][int(line[2])][int(line[1])][0] = 0.5
                    else:
                        pre_images_array[i][int(line[2])][int(line[1])][1] = 0.5

    else:
        pre_images_array = np.zeros((int(len(images_lines)), 256, 256, channels), dtype='float16')
        for i in tqdm(range(0, len(images_lines)), ascii=True):
            incre = (image_label_idx[i + 1] - image_label_idx[i]) // channels
            for j in range(image_label_idx[i], image_label_idx[i + 1]):
                line = read1line.seek1line(f, lines, j)
                if not fixed:
                    if (j - image_label_idx[i]) // incre == channels:

                        """
                        Some index may not be divisible by n_channel, 
                        thus can raise index error, 
                        so we simply put them in the last channel.
                        """

                        pre_images_array[i][int(line[2])][int(line[1])][channels - 1] = \
                            min(pre_images_array[i][int(line[2])][int(line[1])][channels - 1] + 0.1, 1) \
                            if int(line[3]) == 1 else \
                            max(pre_images_array[i][int(line[2])][int(line[1])][channels - 1] - 0.1, -1)
                    else:
                        pre_images_array[i][int(line[2])][int(line[1])][(j - image_label_idx[i]) // incre] = \
                            min(pre_images_array[i][int(line[2])][int(line[1])][(j - image_label_idx[i]) // incre] + 0.1, 1)\
                            if int(line[3]) == 1 else \
                            max(pre_images_array[i][int(line[2])][int(line[1])][(j - image_label_idx[i]) // incre] - 0.1, -1)
                else:
                    if (j - image_label_idx[i]) // incre == channels:
                        pre_images_array[i][int(line[2])][int(line[1])][channels - 1] = 0.5
                    else:
                        pre_images_array[i][int(line[2])][int(line[1])][(j - image_label_idx[i]) // incre] = 0.5

    f.close()
    return pre_images_array


def get_dataset(channels, roots, fixed=False):
    for root in roots:
        post_images_dir = 'data/' + root + '_post.npy'
        pre_images_dir = 'data/' + root + '_pre_' + str(channels) + 'ch' + ('_fixed' if fixed else '') + '.npy'

        root = 'dataset/' + root + '/'
        images_lines, image_label_idx, post_images_array, lines = aps_image_processing(root)
        np.save(post_images_dir, post_images_array)

        pre_images_array = peseudo_image_processing(root, images_lines, image_label_idx, lines, channels, fixed=fixed)
        np.save(pre_images_dir, pre_images_array)
        print(root, 'process done')


def generate_dataset(channels, fixed=False):
    if not os.path.exists('data'):
        os.mkdir('data')

    roots = os.listdir('dataset')
    # if 'README.md' in roots:
    #     roots.remove('README.md')

    get_dataset(channels, roots, fixed=fixed)
    pre_images_dir = 'data/' + roots[0] + '_pre_' + str(channels) + 'ch' + ('_fixed' if fixed else '') + '.npy'
    post_images_dir = 'data/' + roots[0] + '_post.npy'

    pre_images = np.load(pre_images_dir)
    post_images = np.load(post_images_dir)
    length = len(post_images)

    train_data, test_data = pre_images[:int(0.7 * length)], pre_images[int(0.7 * length):]
    train_label, test_label = post_images[:int(0.7 * length)], post_images[int(0.7 * length):]
    os.remove(pre_images_dir)
    os.remove(post_images_dir)

    for root in roots[1:]:
        pre_images_dir = 'data/' + root + '_pre_' + str(channels) + 'ch' + ('_fixed' if fixed else '') + '.npy'
        post_images_dir = 'data/' + root + '_post.npy'

        pre_images = np.load(pre_images_dir)
        post_images = np.load(post_images_dir)
        length = len(pre_images)

        train_data = np.append(train_data, pre_images[:int(0.7 * length)], axis=0)
        train_label = np.append(train_label, post_images[:int(0.7 * length)], axis=0)
        test_data = np.append(test_data, pre_images[int(0.7 * length):], axis=0)
        test_label = np.append(test_label, post_images[int(0.7 * length):], axis=0)

        os.remove(pre_images_dir)
        os.remove(post_images_dir)

    np.save('data/train_' + str(channels) + 'ch' + ('_fixed' if fixed else '') + '.npy', train_data)
    np.save('data/train_label.npy', train_label)

    np.save('data/test_' + str(channels) + 'ch' + ('_fixed' if fixed else '') + '.npy', test_data)
    np.save('data/test_label.npy', test_label)
    print('train data & test data saved, check your \'data/\' directory.')
