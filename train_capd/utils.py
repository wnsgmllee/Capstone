from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    inputs, labels, fnames = zip(*batch)
    return default_collate(inputs), default_collate(labels), fnames
