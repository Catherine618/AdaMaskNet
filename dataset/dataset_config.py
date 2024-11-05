dataset_classes = {
    "PAMAP2": 12,
    "OPPORTUNITY": 17,
    "USC_HAD": 12,
    "UniMiB_SHAR": 17,
    "UCI_HAR": 6,
    "UCI_HAR_2": 6,
    "WISDM": 6,
}

label_to_name = {
    "UCI_HAR": {
        0: 'Walking',
        1: 'Walking Upstairs',
        2: 'Walking Downstairs',
        3: 'Sitting',
        4: 'Standing',
        5: 'Laying'
    },

    "PAMAP2": {0: 'rope_jumping',
               1: 'lying',
               2: 'sitting',
               3: 'standing',
               4: 'walking',
               5: 'running',
               6: 'cycling',
               7: 'Nordic_walking',
               8: 'ascending_stairs',
               9: 'descending_stairs',
               10: 'vacuum_cleaning',
               11: 'ironing'},
    "UniMiB_SHAR": {
        0: 'StandingUpFS',
        1: 'StandingUpFL',
        2: 'Walking',
        3: 'Running',
        4: 'GoingUpS',
        5: 'Jumping',
        6: 'GoingDownS',
        7: 'LyingDownFS',
        8: 'SittingDown',
        9: 'FallingForw',
        10: 'FallingRight',
        11: 'FallingBack',
        12: 'HittingObstacle',
        13: 'FallingWithPS',
        14: 'FallingBackSC',
        15: 'Syncope',
        16: 'FallingLeft'
    },
    "WISDM":{
        0: 'Sitting',
        1: 'Downstairs',
        2: 'Standing',
        3: 'Walking',
        4: 'Upstairs',
        5: 'Jogging'
    }
}

dataset_parameters = {
    "UCI_HAR": {
        "temporal": [5 * 161 * 1, 10 * (12 * 5) * 161],
        "opt_small_kenerl":(6,1),
        "opt_stride":(1,3),
        "average_size": (9,1)
    },
    "PAMAP2": {
        "temporal": [5 * 239 * 1, 9 * (14 * 5) * 239],
        "opt_small_kenerl":(1,3),
        "opt_stride":(1,2),
        "average_size": (9,1)
    },
    "UniMiB_SHAR": {
        "temporal": [5 * 198 * 1, 10 * (13 * 5) * 198],
        "opt_small_kenerl": (1, 6),
        "opt_stride": (1, 1),
        "average_size": (9,1)
    },
    "WISDM": {
        "temporal": [4 * 329 * 1, 8 * (16 * 4) * 329],
        "opt_small_kenerl": (1, 8),
        "opt_stride": (1, 1),
        "average_size": (9,1)
    }
}
