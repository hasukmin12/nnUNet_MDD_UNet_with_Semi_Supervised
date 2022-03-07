import nibabel as nib
import os
import numpy as np
import shutil


path = '/home/has/Datasets/_has_Task124_Urinary'
urinary_list = next(os.walk(path))[1]
urinary_list.sort()
# print(kid_blad_list)
# print(ureter_list)



for case in urinary_list:
    # print(case)

    urinary_path = os.path.join(path,case,'segmentation.nii.gz')
    # print(urinary_path)

    urinary_mask = nib.load(urinary_path).get_fdata()


    if urinary_mask.max() != 3:
        print(case)
        print(urinary_mask.max())

    # z_axis = int(urinary_mask.shape[0])
    #
    #
    # for z in range(0, z_axis):
    #     for x in range(0, 512):
    #         for y in range(0, 512):
    #             if urinary_mask[z][x][y] == 4:
    #                 urinary_mask[z][x][y] = 3




