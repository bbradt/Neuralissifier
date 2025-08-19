from neuralissifier.dataloaders.fake_fnc import FakeFNC
# from dataloaders.load_dev_data import UKBData
# from dataloaders.load_UKB_HCP1200 import UKBHCP1200Data
# from dataloaders.cadasil import CadasilData
from neuralissifier.dataloaders.CSVDataset import CSVDataset
from neuralissifier.dataloaders.CSVDataset_preload import RAMDataset


def get_ramset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    # elif key.lower() == "ukb":
    #    return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_under4mdd":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          temporal_idx=[1,   4,   6,   9,  12,  14,  17,  20,  23,  25,  28,  31,  33,
                                        36,  39,  42,  44,  47,  50,  53,  55,  58,  61,  63,  66,  69,
                                        72,  74,  77,  80,  82,  85,  88,  91,  93,  96,  99, 101, 104,
                                        107, 110, 112, 115, 118, 121, 123, 126, 129, 131, 134, 137, 140,
                                        142, 145, 148, 150, 153, 156, 159, 161, 164, 167, 170, 172, 175,
                                        178, 180, 183, 186, 189, 191, 194, 197, 199, 202, 205, 208, 210,
                                        213, 216, 219, 221, 224, 227, 229, 232, 235, 238, 240, 243, 246,
                                        248, 251, 254, 257, 259, 262, 265, 267, 270, 273, 276, 278, 281,
                                        284, 287, 289, 292, 295, 297, 300, 303, 306, 308, 311, 314, 316,
                                        319, 322, 325, 327, 330, 333, 336, 338, 341, 344, 346, 349, 352,
                                        355, 357, 360, 363, 365, 368, 371, 374, 376, 379, 382, 384, 387,
                                        390, 393, 395, 398, 401, 404, 406, 409, 412, 414, 417, 420, 423,
                                        425, 428, 431, 433, 436, 439, 442, 444, 447],
                          **kwargs
                          )
    elif key.lower() == "ukbhcp_v2_smri":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_smri":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_sfnc":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_sfnc":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_tc_static":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_static.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_tc_static":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_spectrogram":
        return RAMDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/spectrograms_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbonly":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="UKB",
                          **kwargs)
    elif key.lower() == "hcponly":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCP",
                          **kwargs)
    elif key.lower() == "hcpaonly":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCP_Aging",
                          **kwargs)
    elif key.lower() == "hcpdonly":
        return RAMDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCPD",
                          **kwargs)
    # elif key.lower() == "cadasil":
    #    return CadasilData(*args, **kwargs)


def get_dataset(key, *args, **kwargs):
    if key.lower() == "fakefnc":
        return FakeFNC(*args, **kwargs)
    # elif key.lower() == "ukb":
    #    return UKBData(*args, **kwargs)
    elif key.lower() == "ukbhcp":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_smri":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_smri":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/smri_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_sfnc":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_sfnc":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/sfnc_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_tc_static":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_static.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v3_tc_static":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/tc_converted_files_v3.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_spectrogram":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/spectrograms_v2.csv",
                          **kwargs)
    elif key.lower() == "ukbhcp_v2_under4mdd":
        return CSVDataset(*args,
                          age_csv="/data/users3/mduda/scripts/brainAge/HCP_HCPA_UKB_age_filepaths_dFNC_sMRI_cogScores_v2.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v2.csv",
                          temporal_idx=[0,   3,   5,   8,  11,  13,  16,  19,  22,  24,  27,  30,  32,
                                        35,  38,  41,  43,  46,  49,  52,  54,  57,  60,  62,  65,  68,
                                        71,  73,  76,  79,  81,  84,  87,  90,  92,  95,  98, 100, 103,
                                        106, 109, 111, 114, 117, 120, 122, 125, 128, 130, 133, 136, 139,
                                        141, 144, 147, 149, 152, 155, 158, 160, 163, 166, 169, 171, 174,
                                        177, 179, 182, 185, 188, 190, 193, 196, 198, 201, 204, 207, 209,
                                        212, 215, 218, 220, 223, 226, 228, 231, 234, 237, 239, 242, 245,
                                        247, 250, 253, 256, 258, 261, 264, 266, 269, 272, 275, 277, 280,
                                        283, 286, 288, 291, 294, 296, 299, 302, 305, 307, 310, 313, 315,
                                        318, 321, 324, 326, 329, 332, 335, 337, 340, 343, 345, 348, 351,
                                        354, 356, 359, 362, 364, 367, 370, 373, 375, 378, 381, 383, 386,
                                        389, 392, 394, 397, 400, 403, 405, 408, 411, 413, 416, 419, 422,
                                        424, 427, 430, 432, 435, 438, 441, 443, 446],
                          **kwargs
                          )
    elif key.lower() == "ukbonly":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="UKB",
                          **kwargs)
    elif key.lower() == "hcponly":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCP",
                          **kwargs)
    elif key.lower() == "hcpaonly":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCP_Aging",
                          **kwargs)
    elif key.lower() == "hcpdonly":
        return CSVDataset(*args,
                          age_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/HCP_HCPA_HCPD_UKB_age_filepaths_dFNC_sMRI_cogScores_v3.csv",
                          converted_csv="/data/users3/bbaker/projects/Dynamic_BrainAge/data/converted_files_v3.csv",
                          study="HCPD",
                          **kwargs)
    # elif key.lower() == "cadasil":
    #    return CadasilData(*args, **kwargs)


if __name__ == "__main__":
    dataset = get_ramset(
        "hcponly",
        N_subs=999999999,
        sequential=False,
        N_timepoints=448
    )
    print(len(dataset.subjects))
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[50][0].shape)
    #print(dataset[5000][0].shape)
    #print(dataset[10000][0].shape)
    #print(dataset[15000][0].shape)
    print(dataset[-1][0].shape)
