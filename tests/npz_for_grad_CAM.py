import os
import numpy as np
import nibabel as nib

path = '/data/sukmin/CT_2_inference_rst_130_test_e250/CT_240.npz'
vol_numpy = np.load(path)
print(vol_numpy['f']['softmax'])

# xform = np.eye(4) * 2
# label_Nifti = nib.nifti1.Nifti1Image(vol_numpy, xform)
#
#
# nib.save(label_Nifti, output_path)
# print("save nii")