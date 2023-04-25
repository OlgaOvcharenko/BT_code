import os
import pathlib
import platform

from hdfs import InsecureClient


def is_in_cluster():
    if platform.system() == 'Windows':
        return False
    else:
        hs = os.uname()[1]
        return hs == 'tango'


def get_full_path_str(path: str):
    print("file://////" + str(pathlib.Path().absolute()) + "/" + path)
    return "file://////" + str(pathlib.Path().absolute()) + "/" + path
    # return str(pathlib.Path().absolute()) + "/" + path


def write_to_hdfs():
    hostname = "charlie.dm.isds.tugraz.at"
    port = 9870
    hdfs_client = InsecureClient(f"http://{hostname}:{port}")

    return hdfs_client
