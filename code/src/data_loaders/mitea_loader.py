import os
import glob
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

class MITEADataset(Dataset):
    def __init__(self, data_dir, split='train', num_sparse_slices=3, transform=None):
        """
        Dataset for MITEA data, simulating sparse 2D slice inputs for 4D reconstruction.
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.labels_dir = os.path.join(data_dir, 'labels')
        self.num_sparse_slices = num_sparse_slices
        self.transform = transform
        
        # Find subjects
        all_files = glob.glob(os.path.join(self.images_dir, 'MITEA_*_scan1_ED.nii.gz'))
        subject_ids = sorted([os.path.basename(f).split('_scan')[0] for f in all_files])
        
        # Simple split (can be improved later)
        if split == 'train':
            self.subject_ids = subject_ids[:110]
        elif split == 'val':
            self.subject_ids = subject_ids[110:125]
        else:
            self.subject_ids = subject_ids[125:]

    def __len__(self):
        return len(self.subject_ids)

    def _normalize(self, data):
        data = data.astype(np.float32)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val > 0:
            data = (data - min_val) / (max_val - min_val)
        return data

    def __getitem__(self, idx):
        subject = self.subject_ids[idx]
        
        # Load Scan 1 ED and ES
        img_ed_path = os.path.join(self.images_dir, f"{subject}_scan1_ED.nii.gz")
        img_es_path = os.path.join(self.images_dir, f"{subject}_scan1_ES.nii.gz")
        
        # Load labels for validation/supervision
        lbl_ed_path = os.path.join(self.labels_dir, f"{subject}_scan1_ED.nii.gz")
        lbl_es_path = os.path.join(self.labels_dir, f"{subject}_scan1_ES.nii.gz")
        
        img_ed_nii = nib.load(img_ed_path)
        img_es_nii = nib.load(img_es_path)
        lbl_ed_nii = nib.load(lbl_ed_path)
        
        img_ed = self._normalize(img_ed_nii.get_fdata())
        img_es = self._normalize(img_es_nii.get_fdata())
        lbl_ed = lbl_ed_nii.get_fdata().astype(np.float32)
        
        # To simulate sparse slices, we extract slices along different axes
        # (Assuming the volume is roughly centered on the heart)
        h, w, d = img_ed.shape
        
        slices_ed = []
        slices_es = []
        
        # Extract middle slices in each dimension
        # In a more advanced version, we'd use the affine to find specific planes (4CH, 2CH, SAX)
        slices_ed.append(img_ed[h//2, :, :])
        slices_ed.append(img_ed[:, w//2, :])
        slices_ed.append(img_ed[:, :, d//2])
        
        slices_es.append(img_es[h//2, :, :])
        slices_es.append(img_es[:, w//2, :])
        slices_es.append(img_es[:, :, d//2])

        return {
            'subject': subject,
            'sparse_slices_ed': [torch.from_numpy(s).float() for s in slices_ed],
            'sparse_slices_es': [torch.from_numpy(s).float() for s in slices_es],
            'full_volume_ed': torch.from_numpy(img_ed).float(),
            'full_volume_es': torch.from_numpy(img_es).float(),
            'occupancy_ed': torch.from_numpy(lbl_ed).float(),
            'affine': torch.from_numpy(img_ed_nii.affine).float()
        }
