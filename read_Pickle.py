import pickle
from collections import OrderedDict as Odict

path = '/data5/sukmin/nnunet_process_out/Task257_Ureter/splits_final.pkl'

# path = '/data5/sukmin/splits_final.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(data)

# # save
# with open('splits_final.pkl', 'wb') as f:
#     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

