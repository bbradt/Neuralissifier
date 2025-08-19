import os
import glob
from bids import BIDSLayout
import pandas as pd
import nibabel as nib
from bids_validator import BIDSValidator
from torch.utils.data import Dataset


class NeuroimagingDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 bids_path: str,
                 subject_key: str,
                 modality_keys: list = ['anat'],
                 datafile_keys: list = ['Sm6mwc1pT1.nii'],
                 data_keys: list = ['filename'],
                 label_keys: list = ['sDEMOG_DIAGNOSIS']):
        """
            Generic Neuroimaging Dataset object, which is supe
        """
        if not os.path.exists(csv_path):
            raise (Exception("The CSV %s does not exist." % csv_path))
        if not os.path.exists(bids_path):
            raise (Exception("The BIDS path %s does not exist." % bids_path))
        self.labels = pd.read_csv(csv_path, dtype={subject_key: "string"})
        self.subject_key = subject_key
        self.modality_keys = modality_keys
        self.datafile_keys = datafile_keys
        self.data_keys = data_keys
        self.label_keys = label_keys
        self.subjects = []
        validator = BIDSValidator()
        bids_object = None
        if validator.is_bids(bids_path):
            bids_object = BIDSLayout(bids_path)
        if bids_object is None:
            subject_dirs = glob.glob(os.path.join(bids_path,
                                                  "*"))
            rows = []
            for subject_dir in subject_dirs:                
                subject_id = os.path.basename(subject_dir)
                subject_row = self.labels.loc[self.labels[subject_key]
                                              == subject_id]
                if len(subject_row) == 0:
                    print(("No subject with ID %s found" % subject_id))
                    continue
                self.subjects.append(subject_id)
                subject_dict = subject_row.to_dict(orient="records")[-1]
                session_dirs = glob.glob(os.path.join(subject_dir,
                                                      "*"))
                for session_dir in session_dirs:
                    modality_dirs = glob.glob(os.path.join(session_dir,
                                                           "*"))
                    for modality_dir in modality_dirs:
                        files = glob.glob(os.path.join(modality_dir, "*"))
                        modality = os.path.basename(modality_dir)
                        for file in files:
                            if not os.path.isdir(file):
                                filename = os.path.basename(file)
                                row = dict(**subject_dict,
                                           subject_id=subject_id,
                                           modality=modality,
                                           filename=file,
                                           datafile=filename)
                                rows.append(row)
            self.dataset = pd.DataFrame(rows)
        else:
            raise (NotImplementedError(
                "BIDS-compliant data sets not fully implemented."))

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, k):
        subject = self.subjects[k]
        subject_row = self.dataset[self.dataset[self.subject_key]==subject]
        data_list = []
        label_list = []
        for modality_key in self.modality_keys:
            subject_row = subject_row[subject_row['modality'] == modality_key]
            for datafile_key in self.datafile_keys:
                subject_row = subject_row[subject_row['datafile'] == datafile_key]
                for data_key in self.data_keys:
                    value = subject_row[data_key].values[0]
                    if '.nii' in value:
                        value = nib.load(value).get_fdata()
                    data_list.append(value)
        for label_key in self.label_keys:       
            value =  subject_row[label_key].values[0]
            value = list(self.dataset[label_key].unique()).index(value)
            label_list.append(value)
        return data_list, label_list


if __name__ == "__main__":
    bids_path = os.path.join('/data',
                             'qneuromark',
                             'Data',
                             'FBIRN',
                             'Data_BIDS',
                             'Raw_Data')
    csv_path = os.path.join('/data',
                            'qneuromark',
                            'Data',
                            'FBIRN',
                            'Data_info',
                            'fBIRN_CMINDS_4rsfMRI2_G.csv')
    nid = NeuroimagingDataset(csv_path, bids_path, 'SubjectID')
    
