import datasets.freemask_semseg as freemask_semseg
import datasets.outdoor_semseg as outdoor_semseg
import datasets.scannet2d3d as scannet2d3d
import datasets.semseg as semseg

DATASETS = []


def add_dataset(module):
    DATASETS.extend([getattr(module, a) for a in dir(module) if "Dataset" in a])

add_dataset(freemask_semseg)
add_dataset(outdoor_semseg)
add_dataset(scannet2d3d)
add_dataset(semseg)

def get_datasets():
    """Returns a tuple of sample datasets."""
    return DATASETS


def load_dataset(name):
    """Creates and returns an instance of the dataset given its class name."""
    # Find the model class from its name
    all_datasets = get_datasets()
    dsdict = {ds.__name__: ds for ds in all_datasets}
    if name not in dsdict:
        print("Invalid model index. Options are:")
        # Display a list of valid model names
        for ds in all_datasets:
            print(f"\t* {ds.__name__}")
        return None
    DSClass = dsdict[name]

    return DSClass