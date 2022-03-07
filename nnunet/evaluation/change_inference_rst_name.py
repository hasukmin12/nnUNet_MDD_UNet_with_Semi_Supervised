import os
import shutil

# GT_path = '/data/sukmin/nnUNet_raw_data_base/nnUNet_raw_data/Task130_Ureter/labelsTs'
inf_path = '/data5/sukmin/CT_2_inference_rst_263_fc'

aim_path = '/data5/sukmin/inf_2_GT_263_fc'
if os.path.isdir(aim_path)==False:
    os.makedirs(aim_path)

# GT_list = next(os.walk(GT_path))[2]
# GT_list.sort()
# print(GT_list)

inf_list = next(os.walk(inf_path))[2]
inf_list.sort()
# print(inf_list)

inf_nii_list = []

for list in inf_list:
    if list[-3:] == '.gz':
        inf_nii_list.append(list)

# print(inf_nii_list)


for case in inf_nii_list:
    print(case[3:6])
    input_path = os.path.join(inf_path,case)
    # output_path = os.path.join(aim_path,'case_{0:05d}.nii.gz'.format(inf_nii_list.index(case)+ 240))
    output_path = os.path.join(aim_path, 'case_{0:05d}.nii.gz'.format(int(case[3:6])))
    print(input_path)
    print(output_path)

    shutil.copyfile(input_path,output_path)