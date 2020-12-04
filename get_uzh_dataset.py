import re
import urllib.request
import os
import zipfile

prev = -1


def download_dataset_by_urls(path):
    url = 'http://rpg.ifi.uzh.ch/davis_data.html'
    website = urllib.request.urlopen(url)
    html = website.read().decode('utf-8')
    links = re.findall(r'datasets/davis/\w+\.zip', html)
    i = 0
    while i < len(links):
        if 'plot' in links[i]:
            del links[i]
            continue
        i += 1

    for i in range(len(links)):
        links[i] = 'http://rpg.ifi.uzh.ch/' + links[i]

    for link in links:
        idx = -1
        while link[idx] != '/':
            idx -= 1
        print('==> Downloading', link[idx + 1:])
        urllib.request.urlretrieve(link, filename=path + '/' + link[idx + 1:], reporthook=schedule)


def schedule(blocknum, blocksize, totalsize):
    global prev
    percent = int(100.0 * blocknum * blocksize / totalsize)
    if percent > 100:
        percent = 100
    if percent % 10 == 0 and (prev == -1 or prev != percent):
        prev = percent
        print("%d%%" % percent)


def unzip_files(path):
    filenames = os.listdir(path)
    for filename in filenames:
        filepath = os.path.join(path, filename)
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            new_file_path = filename.split(".", 1)[0]
            new_file_path = os.path.join(path, new_file_path)
            if os.path.isdir(new_file_path):
                pass
            else:
                os.mkdir(new_file_path)
            zip_ref.extractall(new_file_path)
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    path = 'dataset'
    if not os.path.exists(path):
        os.mkdir(path)
    download_dataset_by_urls(path)
    unzip_files(path)
