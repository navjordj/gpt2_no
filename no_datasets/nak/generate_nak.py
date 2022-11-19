import tarfile
import glob

import tqdm

from lxml import etree

def main():

    for tar_file in tqdm.tqdm(glob.glob("data/*.tar")):
        with tarfile.open(tar_file, 'r') as f:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, "extracted_data/")

    for tar_file in tqdm.tqdm(glob.glob("extracted_data/**/*nob.tar.gz")):
        with tarfile.open(tar_file, 'r') as f:
            try:
                f.extractall("xmls/")
            except:
                print(f"OS error on {tar_file}")

    with open("final_dataset/nak_dataset.txt", "wb") as f:
        for xml_file in tqdm.tqdm(glob.glob("xmls/**/*.xml")):
            try:
                parsed_tree = etree.parse(xml_file)

                p_tags = parsed_tree.findall('.//p')
                for tag in p_tags:
                    f.write((tag.text + "\n").encode("utf-8"))
            except Exception as e:
                pass

# Upload final_dataset/nak_dataset.txt to navjordj/nak_nb


if __name__ == '__main__':
    main()
