TRAIN_CSV = '../data/raw/train.csv'
TEST_CSV = '../data/raw/test.csv'
STAT_CSV = '../data/raw/pixel_stats.csv'
TRAIN_CONTROL_CSV = '../data/raw/train_controls.csv'
TEST_CONTROL_CSV = '../data/raw/test_controls.csv'

SPLIT_TXT = 'split.txt'

SEED = 112

N_CLASS = 1108
N_CLASS_CONTROL = 31
N_CLASS_CONTROL_PLATE = N_CLASS_CONTROL * 4

EXPS = ['HEPG2', 'HUVEC', 'RPE', 'U2OS']
TEST_COUNT = {'HEPG2': 4429, 'HUVEC': 8846, 'RPE': 4417, 'U2OS': 2205}

VALIDATION_EXPS = ('HEPG2-06', 'HUVEC-16', 'RPE-07', 'U2OS-03')

BASE_AUG = [
    ('RandomRotate90', {'p': 0.5}),  # Randomly rotate the input by 90 degrees zero or more times.
    ('Flip', {'p': 0.5}),
    ('Transpose', {'p': 0.5}),
]
