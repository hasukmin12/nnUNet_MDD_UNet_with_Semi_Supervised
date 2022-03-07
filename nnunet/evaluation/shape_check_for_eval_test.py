import nibabel as nib
import os

folder_with_gt = '/data/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task120_Ureter/labelsTs'
folder_with_pred = '/data/sukmin/inf_2_GT'

GT_list = next(os.walk(folder_with_gt))[2]
GT_list.sort()
print(GT_list)

pred_list = next(os.walk(folder_with_pred))[2]
pred_list.sort()
print(pred_list)

for case in GT_list:
    GT_path = os.path.join(folder_with_gt,case)
    pred_path = os.path.join(folder_with_pred,case)
    GT = nib.load(GT_path).get_fdata()
    pred = nib.load(pred_path).get_fdata()

    if GT.shape != pred.shape:
        print(case)
        print(GT.shape)
        print(pred.shape)
        print("")