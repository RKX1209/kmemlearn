import matplotlib
if __name__ == '__main__':
    matplotlib.use("Agg")
from matplotlib import pyplot as plt
import json
import h5py
import numpy as np
import chainer.dataset
import pickle
import pdb

slice_skip = 1

def JsonToVec(json_dict, magic_number=0x1000, magic_start=11, magic_width=3):
    n_columns= magic_number * len(json_dict.keys()) * len(json_dict['SYS_READ'].keys())
    #print("columns: {} = 100k x {} x {}".format(n_columns, len(json_dict.keys()), len(json_dict['SYS_READ'].keys())))
    xi = np.zeros((n_columns,), dtype=np.float32)
    for e, epoch in enumerate(json_dict.keys()):
        for ev, event in enumerate(json_dict[epoch].keys()):
            #if event == 'EVENT_NEAR' or event == 'EVENT_FAR':
            if event != None:
                event_val = json_dict[epoch][event]
                for k, v in event_val.items():
                    ki = int(k[magic_start:magic_start+magic_width],16)
                    #xi[ki + (e * magic_number) + (ev * magic_number)] = 1
                    xi[ki + (e * magic_number) + (ev * magic_number)] += int(v)
    return xi

#@profile
def InitVec(dataset_label, magic_number=0x1000, magic_start=11, magic_width=3):
    X = []
    y = []
    for i in range(0, len(dataset_label) - 1, 2):
        fp = open(dataset_label[i], 'r')
        json_dict = json.load(fp)
        malicious = int(dataset_label[i+1])
        print(malicious, dataset_label[i], len(json_dict.keys()))
        for u in range(0, len(json_dict), slice_skip):
            xi = JsonToVec(json_dict[str(u)], magic_number=magic_number, magic_start=magic_start, magic_width=magic_width)
            X.append(xi)
            y.append(malicious)
        fp.close()
    X = np.vstack(X)
    return X, y

def SliceMerge(X, slice_block = 1):
    min_col = min(X[b::slice_block].shape[0] for b in range(slice_block))
    tmpx = np.zeros((min_col, X.shape[1]), dtype='int32')    
    for b in range(slice_block):
       tmpx = tmpx | np.array(X[b::slice_block][:min_col], dtype='int32')
    return tmpx
                        
class WindowedDataSet(chainer.dataset.DatasetMixin):
    def __init__(self, X, label, time_window_height=20, time_window_skip=1):
        "X: (time_window, memory_addr_bins)"
        self.X = X
        self.label = label
        self.wh = time_window_height
        self.ws = time_window_skip
    def __len__(self):
        return (len(self.X)-self.wh)//self.ws
    def get_example(self, i):
        ib = self.ws*i
        ie = ib+self.wh
        return (self.X[ib:ie, :], self.label)

class BinaryDataSet(chainer.dataset.DatasetMixin):
    def __init__(self, dataset):
        self.ds = dataset
    def get_example(self, i):
        X, Y = self.ds[i]
        X = (X > 0).astype(np.float32)
        return (X, Y)
    def __len__(self):
        return len(self.ds)

class MizumashiDataSet(chainer.dataset.DatasetMixin):
    "just repeat"
    def __init__(self, org_dataset, mizumashi_factor):
        self.ds = org_dataset
        self.ds_len = len(self.ds)
        self.mizumashi_len = int(self.ds_len*mizumashi_factor)
    def __len__(self):
        return self.mizumashi_len
    def get_example(self, i):
        return self.ds.get_example(i%self.ds_len)

class CombinedDataSet(chainer.dataset.DatasetMixin):
    def __init__(self, datasetL):
        self.dsL = datasetL
        self.lenL = [len(ds) for ds in self.dsL]
        return
    def __len__(self):
        return sum(self.lenL)
    def get_example(self, i):
        for li in range(len(self.lenL)):
            if i < 0:
                raise ValueError
            if i < self.lenL[li]:
                return self.dsL[li].get_example(i)
            i -= self.lenL[li]
        raise ValueError("index {} is invalid".format(i))

def load_datasets_pickle(pickle_filename, slice_merge = 1):
    # (index, label) for test and train, with couting
    try:
        with open(pickle_filename, "rb") as fp:
            datasets = pickle.load(fp)
            XLtmp = pickle.load(fp)
    except IOError as e:
        print("--> run python ./prep_data.py to prepare dataset.")
        raise e
    # don't forget to add channel dimension
    XL = []
    for X in XLtmp:
        X = SliceMerge(X, slice_merge)
        xs = X.shape + (1,)
        X = X.reshape(xs)
        XL.append(X)
    del XLtmp
    return (datasets, XL)

def prep_windowed_datasets(pickle_filename, test_dataset_names, train_dataset_names, balance_train_samples=True, window_height=20, window_skip=1, slice_merge=1):
    "train_dataset_names may be empty: then all dataset except specified as test"

    datasets, XL = load_datasets_pickle(pickle_filename, slice_merge)
    print 'pickle:', pickle_filename
    testiL_positives = []
    testiL_negatives = []
    trainiL_positives = []
    trainiL_negatives = []
    for i, (dsname, dslabel) in enumerate(datasets):
        dslabel = np.array([dslabel], dtype=np.int32)[0]
        hit = False
        #print('### ds: {} {} --> test'.format(dsname, dslabel))
        if any(s.endswith(dsname) for s in test_dataset_names):
            print('### ds: {} {} --> test'.format(dsname, dslabel))
            hit = True
            if dslabel == 0:
                testiL_negatives.append((i, dslabel))
            else:
                testiL_positives.append((i, dslabel))
        elif not train_dataset_names or (dsname in train_dataset_names):
            print('### ds: {} {} --> train'.format(dsname, dslabel))
            hit = True
            if dslabel == 0:
                trainiL_negatives.append((i, dslabel))
            else:
                trainiL_positives.append((i, dslabel))
        if not hit:
            print('### ds: {} {} --> ignored'.format(dsname, dslabel))
    print("## train (positive): {}".format([datasets[i][0] for i, label in trainiL_positives]))
    print("## train (negative): {}".format([datasets[i][0] for i, label in trainiL_negatives]))
    print("## test (positive) : {}".format([datasets[i][0] for i, label in testiL_positives]))
    print("## test (negative) : {}".format([datasets[i][0] for i, label in testiL_negatives]))
    test_pos_len_sum = sum([XL[i].shape[0] for i, label in testiL_positives])
    test_neg_len_sum = sum([XL[i].shape[0] for i, label in testiL_negatives])
    print("## test pos:neg = {}:{}".format(test_pos_len_sum, test_neg_len_sum))

    # XXX I feel some room for optimization here -- doi
    testdsL = [WindowedDataSet(XL[i], label, window_height, window_skip) for (i, label) in testiL_positives + testiL_negatives]
    testds = CombinedDataSet(testdsL)
    binaryds = BinaryDataSet(testds)
    #binaryds = testds
    traindsL_positives = [WindowedDataSet(XL[i], label, window_height, window_skip) for (i, label) in trainiL_positives]
    traindsL_negatives = [WindowedDataSet(XL[i], label, window_height, window_skip) for (i, label) in trainiL_negatives]
    trainds_positives = CombinedDataSet(traindsL_positives)
    trainds_negatives = CombinedDataSet(traindsL_negatives)
    if balance_train_samples:
        assert len(trainds_positives) > 0 and len(trainds_negatives) > 0
        if len(trainds_positives) > len(trainds_negatives):
            f = len(trainds_positives)/len(trainds_negatives)
            trainds_negatives = MizumashiDataSet(trainds_negatives, f)
        elif len(trainds_negatives) > len(trainds_positives):
            f = len(trainds_negatives)/len(trainds_positives)
            trainds_positives = MizumashiDataSet(trainds_positives, f)
    print("## train_pos:train_neg = {}:{}".format(len(trainds_positives), len(trainds_negatives)))
    trainds = BinaryDataSet(CombinedDataSet([trainds_positives, trainds_negatives]))
    #trainds = CombinedDataSet([trainds_positives, trainds_negatives])
    return (trainds, binaryds)


def main(fig_out="figures/"):
    import os
    import os.path

    datasets = [
        ("adore", 1),
        ("adore0", 1),
        ("adore1", 1),
        ("adore2", 1),
        ("afkit", 1),
        ("diam", 1),
        ("diam1", 1),
        ("kbeast", 1),
        ("kbeast0", 1),
        ("kbeast1", 1),
        ("kbeast2", 1),
        ("normal", 0),
        ("normal0", 0),
        ("normal1", 0),
        ("normal2", 0),
        ("srootkit", 1),
        ("srootkit0", 1),
        ("srootkit1", 1),
        ("srootkit2", 1),
        ("suterusu", 1),
        ("suterusu0", 1),
        ("suterusu2", 1)
        ]

    dirname = "./data/"
    XL = []

    for fname, label in datasets:
        json_dict = json.load(open(os.path.join(dirname, fname), 'r'))
        X = []
        for u in range(0, len(json_dict), slice_skip):
            xi = JsonToVec(json_dict[str(u)], magic_number=0x1000, magic_start=11, magic_width=3)
            X.append(xi)
        XS = np.vstack(X)
        XL.append(XS)
        print(fname, np.sum(XS)/XS.shape[0], np.max(XS), label)

    # make filtered data
    all_sum = np.zeros(XL[0].shape[1]) # memory axis
    for i in range(len(XL)):
        all_sum += np.sum(XL[i], axis=0)
    select_nz = all_sum > 0
    print("filtered dimension size is {}".format(sum(select_nz)))
    XLF = []
    for i in range(len(XL)):
        XLF.append(XL[i][:,select_nz])
    with open("data.pickle", "wb") as pickle_file:
        pickle.dump(datasets, pickle_file)
        pickle.dump(XL, pickle_file)
        pickle.dump(None, pickle_file) # just keep format order
    with open("data_filtered.pickle", "wb") as pickle_file:
        pickle.dump(datasets, pickle_file)
        pickle.dump(XLF, pickle_file)
        pickle.dump(select_nz, pickle_file)


if __name__ == '__main__':
    main()
    #ds = prep_windowed_datasets("data/data_filtered.pickle", ["suterusu", "suterusu0", "suterusu2"], None)
