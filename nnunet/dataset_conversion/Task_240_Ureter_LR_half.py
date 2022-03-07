#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data


if __name__ == "__main__":
    """
    This is the Bladder dataset from sukmin Ha
    """

    base = "/data5/sukmin/_has_Task240_Ureter_LR_half"

    task_id = 240
    task_name = "Ureter"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/data5/sukmin/nnUNet_raw_data_base/nnUNet_raw_data'

    out_base = join(nnUNet_raw_data, foldername)
    # out_base = join(base, foldername)


    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)

    train_patients = all_cases[:182] + all_cases[251:]
    test_patients = all_cases[182:251]

    # train_patients = all_cases[:210] + all_cases[300:540] + all_cases[600:]
    # test_patients = all_cases[210:300] + all_cases[540:600]


    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    # 나중에 test inference를 위해 폴더는 만들어놓
    for p in test_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        shutil.copy(label_file, join(labelsts, p + ".nii.gz"))


    json_dict = {}
    json_dict['name'] = "Ureter"
    json_dict['description'] = "Ureter segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "Ureter data for nnunet"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "LU_Ureter", #Left UP Ureter
        "2": "RU_Ureter",
        "3": "RD_Ureter",
        "4": "LD_Ureter"

        # "6": "RU_Ureter",
        # "7": "RD_Ureter",
        # "9": "LD_Ureter"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    # json_dict['test'] = [{'image': "./imagesTs/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTs/%s.nii.gz" % i.split("/")[-1]} for i in
    #                    test_patient_names]
    save_json(json_dict, os.path.join(out_base, "dataset.json"))