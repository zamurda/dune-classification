from csvtools.helpers import file_in_dir


class LoadFeatures:
    def __init__(self, filename):
        if not file_in_dir(filename):
            raise FileNotFoundError(f"{filename} doesn't exist in the specified directory")
        else:
            pass


