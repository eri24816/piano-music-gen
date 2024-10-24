


from tqdm import tqdm


if __name__ == '__main__':
    device = 'cuda'

    from pianogen.dataset.pianorolldataset import PianoRollDataset
    from pianogen.dataset.with_feature import FeatureDataset
    from torch.utils.data import DataLoader
    from pianogen.model.with_feature import Cake


    model = Cake(a0_size=512, max_len=32*150, dim_model=256, num_layers=6, num_heads=8, dim_feedforward=1024)

    pr_ds = PianoRollDataset(r'W:\music\music-data-analysis\data', max_duration=32*150) # 150 bars
    ds = FeatureDataset(pr_ds, features=model.features)
    dl = DataLoader(ds,batch_size=8, shuffle=True, num_workers=8)

    for e in range(5):
        for i in tqdm(dl):
            pass