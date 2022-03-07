import os

path = '/data/sukmin/nnunet_process_out/Task064_KiTS/nnUNetData_plans_v2.1_stage0'
path_list = next(os.walk(path))[2]
path_list.sort()

print(path_list)

# for list in path_list:
#     if list[-3:] == 'npy':
#         print(list)
#         os.remove(path+'/'+list)