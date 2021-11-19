import datasets
import numpy as np

class GPT2_no_dataset():

    def __init__(self, datasets_paths, train_size=0.9):
        self.datasets_paths = datasets_paths
        self.train_percentage = str(int(train_size*100)) + "%"

        self.downloaded_datasets = self.download_datasets()

        print(self.downloaded_datasets)
        self.full_dataset = self.concat_datasets()



    def download_datasets(self):
        # TODO: add train/test split
        dataset_list = []
        for path in self.datasets_paths:
            print(f"Downloading {path}")

            if ":" in path:
                split_path = path.split(":")
                path_top = split_path[0]
                path_name = split_path[1]
     
                dl_dataset = datasets.load_dataset(path_top, path_name)["train"]

            else:
                dl_dataset = datasets.load_dataset(path)["train"]
        
            #if "id" not in dl_dataset.features:
            #    print(f"adding id column to {path} dataset")
            #    dl_dataset = dl_dataset.add_column("id", np.arange(len(dl_dataset)))

            if "id" in dl_dataset.features:
                dl_dataset = dl_dataset.remove_columns("id")
            
            dataset_list.append(dl_dataset)

        return dataset_list

    def concat_datasets(self):
        concatenated_dataset = datasets.concatenate_datasets(self.downloaded_datasets)
        return concatenated_dataset


if __name__ == '__main__':

    dataset = GPT2_no_dataset(["navjordj/nak_nb", "oscar:unshuffled_deduplicated_no"])

