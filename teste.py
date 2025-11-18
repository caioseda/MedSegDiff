from vfss_data_split.datasets.vfss_dataset import VFSSImageDataset
import vfss_data_split.data_extraction.video_frame as video_frame
from pathlib import Path

print("Imported VFSSImageDataset from vfss_data_split.datasets.vfss_dataset")

img_path = Path('/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/vfss-data-split/data/dataset_inca')
metadata_csv_path = Path('/Users/caioseda/Documents/Trabalho/Tecgraf/projetos/vfss-data-split/data/metadados/') #video_frame_metadata_test.csv

video_frame_split = {}
dataset_split = {}
for split_name in ['train', 'val', 'test']:
    split_path = metadata_csv_path / f'video_frame_metadata_{split_name}.csv'
    if not split_path.exists():
        print(f"Metadata CSV for {split_name} split does not exist at {split_path}. Skipping.")
        continue

    video_frame_split[split_name] = video_frame.load_video_frame_metadata_from_csv(split_path)
    
    
    dataset = VFSSImageDataset(
        video_frame_df=video_frame_split[split_name],
        target='mask',
        from_images=True
    )
    dataset_split[split_name] = dataset
    print(f"{split_name} dataset length: {len(dataset)}")

print(dataset_split['train'][0])  # Print the first item in the training dataset to verify