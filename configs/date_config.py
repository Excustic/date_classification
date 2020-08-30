from os.path import join
train_labels = ['PR_Class_Model', 'PR_Skin_Model', 'PR_Waste_Model']
train_path = join('split2','train')
valid_path = join('split2','val')
test_path = join('split2','test')
fixed_size = tuple((200, 200))
epochs = 100
sessions = 1
model_name = 'Model_CNN.h5'
weights_path = "weights_best2.hdf5"
batch_size = 32     # larger size might not work on some machines