import os


def download_weights(model_capacitiy):
    from urllib.request import urlretrieve

    weight_file = "crepe-{}.pth".format(model_capacitiy)
    base_url = "https://github.com/sweetcocoa/crepe-pytorch/raw/models/"

    # in all other cases, decompress the weights file if necessary
    package_dir = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(package_dir, weight_file)
    if not os.path.isfile(weight_path):
        print("Downloading weight file {} from {} ...".format(weight_path, base_url + weight_file))
        urlretrieve(base_url + weight_file, weight_path)
