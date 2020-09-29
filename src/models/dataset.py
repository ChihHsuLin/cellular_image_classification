import ujson as json
import numpy as np
import pandas as pd
import re

from albumentations import Compose, Normalize, RandomCrop, Rotate, GaussianBlur
from skimage.filters import gaussian
import torch
from torch.utils.data import Dataset

import constants as c
import util


class CellDataset(Dataset):

    def __init__(self, files, labels, stats, transform, phase, tta=(True, True, True, True), norm='self',
                 gaussian_sigma=0):
        if phase == 'train':
            self.files = []
            self.labels = []
            self.stats = []
            for f, l, s in zip(files, labels, stats):
                self.files.append(f[0])
                self.files.append(f[1])
                self.stats.append([s[0][:6], s[1][:6]])
                self.stats.append([s[0][6:], s[1][6:]])
                self.labels.append(l)
                self.labels.append(l)
        else:
            self.files = files
            self.labels = labels
            self.stats = stats
        self.transform = transform
        self.tta = tta
        self.phase = phase
        self.random = np.random.RandomState(c.SEED)

        self.mean_list = []
        self.std_list = []
        self.norm = norm

        if self.norm == 'exp':
            mean_dict = json.load(open('experiment_mean.json'))
            std_dict = json.load(open('experiment_std.json'))
            for i, f in enumerate(self.files):
                exp, _ = self._get_exp_plate(i)
                self.mean_list.append(mean_dict[exp])
                self.std_list.append(std_dict[exp])
        elif self.norm == 'plate':
            mean_dict = json.load(open('plate_mean.json'))
            std_dict = json.load(open('plate_std.json'))
            for i, f in enumerate(self.files):
                exp, plate = self._get_exp_plate(i)
                key = '%s_%d' % (exp, plate)
                self.mean_list.append(mean_dict[key])
                self.std_list.append(std_dict[key])
        self.gaussian_sigma = gaussian_sigma

    def _get_label(self, idx):
        return self.labels[idx] if self.labels is not None else 0

    def _get_exp_plate(self, idx):
        if self.phase == 'train':
            file_name = self.files[idx]
        else:
            file_name = self.files[idx][0]

        plate = None
        for i in range(1, 5):
            if 'Plate%d' % i in file_name:
                plate = i
                break
        if plate is None:
            raise Exception('Cannot find plate number of {} in SubimageControlDataset'.format(file_name))

        exp = None
        for x in c.EXPS:
            m = re.search("({}-\d+)".format(x), file_name)
            if m:
                exp = m.group(1)
                break
        if exp is None:
            raise Exception('Cannot find experiment number of {} in SubimageControlDataset'.format(file_name))

        return exp, plate

    def __getitem__(self, idx):
        label = self._get_label(idx)
        image = self._read_transform_image(idx)
        return image, label

    def _read_transform_image(self, idx):
        if self.phase == 'train':
            images = [np.load(self.files[idx])]
            stats = [(self.stats[idx][0], self.stats[idx][1])]
        elif self.phase == 'valid':
            images = [np.load(self.files[idx][0]), np.load(self.files[idx][1])]
            stats = [(self.stats[idx][0][:6], self.stats[idx][1][:6]),
                     (self.stats[idx][0][6:], self.stats[idx][1][6:])]
        else:
            images = list()
            raw_images = [np.load(self.files[idx][0]), np.load(self.files[idx][1])]
            # tta
            for i in range(2):
                if self.tta[0]:
                    images.append(raw_images[i])
                if self.tta[1]:
                    images.append(np.fliplr(raw_images[i]))
                if self.tta[2]:
                    images.append(np.flipud(raw_images[i]))
                if self.tta[3]:
                    images.append(np.fliplr(np.flipud(raw_images[i])))

            stats = list()
            stats.extend([(self.stats[idx][0][:6], self.stats[idx][1][:6]),
                          (self.stats[idx][0][6:], self.stats[idx][1][6:])])
            stats.extend([(self.stats[idx][0][:6], self.stats[idx][1][:6])] * 3)
            stats.extend([(self.stats[idx][0][6:], self.stats[idx][1][6:])] * 3)

        ans = []
        for (im_mean, im_std), image in zip(stats, images):
            image = Compose(self.transform)(image=image)['image']
            if self.gaussian_sigma != 0:
                image = image.astype(np.float)
                image[:, :, 0:3] = gaussian(image[:, :, 0:3], sigma=self.gaussian_sigma, multichannel=True)
                image[:, :, 3:6] = gaussian(image[:, :, 3:6], sigma=self.gaussian_sigma, multichannel=True)
            if self.norm == 'self':
                im_mean = np.mean(image, axis=(0, 1))
                im_std = np.std(image, axis=(0, 1)) + 1e-10
                image = Normalize(mean=im_mean, std=im_std, max_pixel_value=1.0)(image=image)['image']
            else:
                image = Normalize(mean=self.mean_list[idx], std=self.std_list[idx], max_pixel_value=1.0)(image=image)[
                    'image']
            ans.append(image)
        if len(ans) > 1:
            images = np.stack(ans, axis=0)
            images = np.transpose(images, (0, 3, 1, 2))
            return images
        else:
            images = np.transpose(ans[0], (2, 0, 1))
            return images

    def __len__(self):
        return len(self.files)


class SubimageDataset(CellDataset):

    def __init__(self, files, labels, stats, transform, phase, tta=(True, True, True, True), norm='self',
                 gaussian_sigma=0):
        super(SubimageDataset, self).__init__(files, labels, stats, transform, phase, tta, norm, gaussian_sigma)

    @staticmethod
    def _get_subimage(image):
        size = image.shape[0] // 2
        return [image[:size, :size, :], image[:size, size:, :], image[size:, :size, :], image[size:, size:, :]]

    def _get_train_subimage(self, idx):
        sub = np.random.randint(4)
        return [SubimageDataset._get_subimage(np.load(self.files[idx]))[sub]]

    def _read_transform_image(self, idx):
        if self.phase == 'train':
            full_image = np.load(self.files[idx])
            full_mean = [np.mean(full_image, axis=(0, 1))]
            full_std = [np.std(full_image, axis=(0, 1))]
            sub = np.random.randint(4)
            images = [SubimageDataset._get_subimage(full_image)[sub]]
            # images = self._get_train_subimage(idx)
        elif self.phase == 'valid':
            full_images = [np.load(self.files[idx][0]), np.load(self.files[idx][1])]
            images = SubimageDataset._get_subimage(full_images[0]) + SubimageDataset._get_subimage(full_images[1])
            full_mean = [np.mean(full_images[0], axis=(0, 1))] * 4 + [np.mean(full_images[1], axis=(0, 1))] * 4
            full_std = [np.std(full_images[0], axis=(0, 1))] * 4 + [np.std(full_images[1], axis=(0, 1))] * 4
        else:
            full_images = [np.load(self.files[idx][0]), np.load(self.files[idx][1])]
            raw_mean = [np.mean(full_images[0], axis=(0, 1)), np.mean(full_images[1], axis=(0, 1))]
            raw_std = [np.std(full_images[0], axis=(0, 1)), np.std(full_images[1], axis=(0, 1))]
            full_mean = list()
            full_std = list()
            images = list()
            raw_images = SubimageDataset._get_subimage(full_images[0]) + SubimageDataset._get_subimage(full_images[1])
            # tta
            for i in range(8):
                if self.tta[0]:
                    images.append(raw_images[i])
                    full_mean.append(raw_mean[i // 4])
                    full_std.append(raw_std[i // 4])
                if self.tta[1]:
                    images.append(np.fliplr(raw_images[i]))
                    full_mean.append(raw_mean[i // 4])
                    full_std.append(raw_std[i // 4])
                if self.tta[2]:
                    images.append(np.flipud(raw_images[i]))
                    full_mean.append(raw_mean[i // 4])
                    full_std.append(raw_std[i // 4])
                if self.tta[3]:
                    images.append(np.fliplr(np.flipud(raw_images[i])))
                    full_mean.append(raw_mean[i // 4])
                    full_std.append(raw_std[i // 4])

        ans = []
        for i, image in enumerate(images):
            image = Compose(self.transform)(image=image)['image']
            if self.gaussian_sigma != 0:
                image = image.astype(np.float)
                image[:, :, 0:3] = gaussian(image[:, :, 0:3], sigma=self.gaussian_sigma, multichannel=True)
                image[:, :, 3:6] = gaussian(image[:, :, 3:6], sigma=self.gaussian_sigma, multichannel=True)
            if self.norm == 'self':
                im_mean = np.mean(image, axis=(0, 1))
                im_std = np.std(image, axis=(0, 1)) + 1e-10
                image = Normalize(mean=im_mean, std=im_std, max_pixel_value=1.0)(image=image)['image']
            elif self.norm == 'image':
                image = Normalize(mean=full_mean[i], std=full_std[i], max_pixel_value=1.0)(image=image)['image']
            else:
                image = Normalize(mean=self.mean_list[idx], std=self.std_list[idx], max_pixel_value=1.0)(image=image)[
                    'image']
            ans.append(image)
        if len(ans) > 1:
            images = np.stack(ans, axis=0)
            images = np.transpose(images, (0, 3, 1, 2))
            return images
        else:
            images = np.transpose(ans[0], (2, 0, 1))
            return images


class Subimage128Dataset(CellDataset):

    def __init__(self, files, labels, stats, transform, phase, tta=(True, True, True, True), norm='self'):
        super(Subimage128Dataset, self).__init__(files, labels, stats, transform, phase, tta, norm)
        if phase == 'train':
            self.transform += [
                # Rotate(limit=90, p=0.8),
                RandomCrop(128, 128, always_apply=True, p=1),
                # GaussianBlur(p=0.8)
            ]

    @staticmethod
    def _get_subimage(image):
        size = image.shape[0] // 4
        image_list = list()
        for i in range(4):
            i_off = size * i
            for j in range(4):
                j_off = size * j
                image_list.append(image[i_off:i_off + size, j_off:j_off + size, :])
        return image_list

    def _read_transform_image(self, idx):
        if self.phase == 'train':
            images = [np.load(self.files[idx])]
        elif self.phase == 'valid':
            images = Subimage128Dataset._get_subimage(np.load(self.files[idx][0])) + Subimage128Dataset._get_subimage(
                np.load(self.files[idx][1]))
        else:
            images = list()
            raw_images = Subimage128Dataset._get_subimage(
                np.load(self.files[idx][0])) + Subimage128Dataset._get_subimage(
                np.load(self.files[idx][1]))
            # tta
            for i in range(len(raw_images)):
                if self.tta[0]:
                    images.append(raw_images[i])
                if self.tta[1]:
                    images.append(np.fliplr(raw_images[i]))
                if self.tta[2]:
                    images.append(np.flipud(raw_images[i]))
                if self.tta[3]:
                    images.append(np.fliplr(np.flipud(raw_images[i])))

        ans = []
        for image in images:
            image = Compose(self.transform)(image=image)['image']
            if self.norm == 'self':
                im_mean = np.mean(image, axis=(0, 1))
                im_std = np.std(image, axis=(0, 1)) + 1e-10
                image = Normalize(mean=im_mean, std=im_std, max_pixel_value=1.0)(image=image)['image']
            else:
                image = Normalize(mean=self.mean_list[idx], std=self.std_list[idx], max_pixel_value=1.0)(image=image)[
                    'image']
            ans.append(image)
        if len(ans) > 1:
            images = np.stack(ans, axis=0)
            images = np.transpose(images, (0, 3, 1, 2))
            return images
        else:
            images = np.transpose(ans[0], (2, 0, 1))
            return images


class SubimageListDataset(SubimageDataset):

    def __init__(self, files, labels, stats, transform, phase, tta=(True, True, True, True), norm='self'):
        super(SubimageListDataset, self).__init__(files, labels, stats, transform, phase, tta, norm)

    def _get_train_subimage(self, idx):
        return SubimageDataset._get_subimage(np.load(self.files[idx]))

    def _get_label(self, idx):
        label = self.labels[idx] if self.labels is not None else 0
        if self.phase == 'train':
            return np.array([label, label, label, label])
        elif self.phase == 'valid':
            return np.array([label, label, label, label, label, label, label, label])
        else:
            return label


class SubimageControlDataset(CellDataset):

    def __init__(self, files, labels, stats, files_ct, labels_ct, stats_ct, transform, phase,
                 tta=(True, True, True, True), norm='self'):
        super(SubimageControlDataset, self).__init__(files, labels, stats, transform, phase, tta, norm)
        if phase == 'train':
            self.files_ct = []
            self.labels_ct = []
            self.stats_ct = []
            for f, l, s in zip(files_ct, labels_ct, stats_ct):
                self.files_ct.append(f[0])
                self.files_ct.append(f[1])
                self.stats_ct.append([s[0][:6], s[1][:6]])
                self.stats_ct.append([s[0][6:], s[1][6:]])
                self.labels_ct.append(l)
                self.labels_ct.append(l)
        else:
            self.files_ct = files_ct
            self.labels_ct = labels_ct
            self.stats_ct = stats_ct

    def __getitem__(self, idx):
        label = self._get_label(idx)
        image = self._read_transform_image(idx)
        exp, plate = self._get_exp_plate(idx)
        image2, label2 = self._get_control_from_exp_plate(exp, plate)
        return image, label, image2, label2

    def _get_label(self, idx, exp_type='treatment'):
        if exp_type == 'treatment':
            labels = self.labels
        elif exp_type == 'control':
            labels = self.labels_ct
        else:
            raise Exception("%s exp_type has to be {'treatment', 'control'}" % exp_type)
        return labels[idx] if labels is not None else 0

    def _get_control_from_exp_plate(self, exp, plate):
        if self.phase == 'train':
            files_ct = self.files_ct
        else:
            files_ct = [x[0] for x in list(self.files_ct)]
        match_f = [f for f in files_ct if exp in f and plate in f]
        f = np.random.choice(match_f)
        idx = files_ct.index(f)
        label = self._get_label(idx, 'control')
        image = self._read_transform_image(idx, 'control')
        return image, label

    @staticmethod
    def _get_subimage(image):
        size = image.shape[0] // 2
        return [image[:size, :size, :], image[:size, size:, :], image[size:, :size, :], image[size:, size:, :]]

    def _get_train_subimage(self, idx, exp_type='treatment'):
        sub = np.random.randint(4)
        if exp_type == 'treatment':
            files = self.files
        elif exp_type == 'control':
            files = self.files_ct
        else:
            raise Exception("%s exp_type has to be {'treatment', 'control'}" % exp_type)
        return [SubimageControlDataset._get_subimage(np.load(files[idx]))[sub]]

    def _read_transform_image(self, idx, exp_type='treatment'):
        if exp_type == 'treatment':
            files = self.files
        elif exp_type == 'control':
            files = self.files_ct
        else:
            raise Exception("%s exp_type has to be {'treatment', 'control'}" % exp_type)

        if self.phase == 'train':
            images = self._get_train_subimage(idx, exp_type)
        elif self.phase == 'valid':
            images = SubimageControlDataset._get_subimage(
                np.load(files[idx][0])) + SubimageControlDataset._get_subimage(
                np.load(files[idx][1]))
        else:
            images = list()
            raw_images = SubimageControlDataset._get_subimage(
                np.load(self.files[idx][0])) + SubimageControlDataset._get_subimage(
                np.load(self.files[idx][1]))
            # tta
            for i in range(8):
                if self.tta[0]:
                    images.append(raw_images[i])
                if self.tta[1]:
                    images.append(np.fliplr(raw_images[i]))
                if self.tta[2]:
                    images.append(np.flipud(raw_images[i]))
                if self.tta[3]:
                    images.append(np.fliplr(np.flipud(raw_images[i])))

        ans = []
        for image in images:
            image = Compose(self.transform)(image=image)['image']
            if self.norm == 'self':
                im_mean = np.mean(image, axis=(0, 1))
                im_std = np.std(image, axis=(0, 1)) + 1e-10
                image = Normalize(mean=im_mean, std=im_std, max_pixel_value=1.0)(image=image)['image']
            else:
                image = Normalize(mean=self.mean_list[idx], std=self.std_list[idx], max_pixel_value=1.0)(image=image)[
                    'image']
            ans.append(image)
        if len(ans) > 1:
            images = np.stack(ans, axis=0)
            images = np.transpose(images, (0, 3, 1, 2))
            return images
        else:
            images = np.transpose(ans[0], (2, 0, 1))
            return images


class PairDataset(SubimageDataset):

    def __init__(self, files, labels, stats, transform, phase, tta=(True, True, True, True)):
        super(PairDataset, self).__init__(files, labels, stats, transform, phase, tta)
        assert phase == 'valid'
        g2rna, masks = util.get_g2rna()
        plate2group = pd.read_csv('plate_group.csv', index_col='plate').to_dict()['group']
        self.mask = []
        for i in range(len(self.files)):
            exp, plate = self._get_exp_plate(i)
            key = '%s_%d' % (exp, plate)
            self.mask.append(masks[plate2group[key]])

    def __getitem__(self, idx):
        images = self._read_transform_image(idx)
        label = self._get_label(idx)
        return images[:4], images[4:], label, self.mask[idx]


class ActDataset(Dataset):

    def __init__(self, features, labels):
        feat_mean = np.mean(features, axis=1)[:, np.newaxis]
        feat_std = np.std(features, axis=1)[:, np.newaxis] + 1e-10
        self.features = (features - feat_mean) / feat_std
        self.labels = labels

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx] if self.labels is not None else None

    def __len__(self):
        return self.features.shape[0]


class ActStatSampler:

    def __init__(self, mean_features, features, labels, p=0.5):
        # p: probability to use mean feature
        self.random = np.random.RandomState(seed=c.SEED)
        self.p = p

        feat_mean = np.mean(mean_features, axis=1)[:, np.newaxis]
        feat_std = np.std(mean_features, axis=1)[:, np.newaxis] + 1e-10
        self.mean_features = (mean_features - feat_mean) / feat_std

        feat_mean = np.mean(features, axis=1)[:, np.newaxis]
        feat_std = np.std(features, axis=1)[:, np.newaxis] + 1e-10
        self.features = (features - feat_mean) / feat_std
        self.labels = labels
        self.class2index = [[] for _ in range(mean_features.shape[0])]
        for i, l in enumerate(labels):
            self.class2index[l].append(i)

    def next(self):
        mean_features = np.zeros_like(self.mean_features)
        for i in range(mean_features.shape[0]):
            if self.random.uniform() < self.p:
                mean_features[i] = self.mean_features[i]
            else:
                index = self.random.choice(self.class2index[i], 1)
                mean_features[i] = self.features[index]
        return mean_features


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data1, self.next_data2 = next(self.loader)
        except StopIteration:
            self.next_data1 = None
            self.next_data2 = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data1 = self.next_data1.cuda(non_blocking=True)
            self.next_data2 = self.next_data2.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data1 = self.next_data1
        data2 = self.next_data2
        self.preload()
        return data1, data2

