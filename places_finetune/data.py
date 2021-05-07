import numpy as np
from torch.utils.data import Dataset
import os.path as osp
# from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize as imresize
import os
from config import places_root, doom_root

class DoomImageDummy(Dataset):
    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return np.zeros((168, 168, 3)), np.eye(4)[0]

class DoomImage(Dataset):
    def __init__(self, train=True, augment=False, permute=True):

        dat = np.load(doom_root)
        im, labels = dat['obs'], dat['label']

        n = im.shape[0]
        split_idx = int(0.9 * n)

        if train:
            im = im[:split_idx]
            labels = labels[:split_idx]
        else:
            im = im[split_idx:]
            labels = labels[split_idx:]

        im, labels = self.rebalance_labels(im, labels, train=train)

        if permute:
            perm_mask = np.random.permutation(im.shape[0])
            im = im[perm_mask]
            labels = labels[perm_mask]

        self.data = im
        self.labels = labels

        self.one_hot_map = np.eye(4)

        self.train = train

    def __len__(self):
        return self.data.shape[0]

    def rebalance_labels(self, data, labels, train=False):
        data_list = []
        labels_list = []
        counts = []

        for i in range(4):
            count_i = (labels == i).sum()
            counts.append(count_i)

        counts = np.array(counts)
        min_count = counts.min()

        for i in range(4):
            label_mask = (labels == i)
            data_mask = data[label_mask]
            label_mask = labels[label_mask]
            data_list.append(data_mask[:min_count])
            labels_list.append(label_mask[:min_count])

        data_list = np.concatenate(data_list, axis=0)
        labels_list = np.concatenate(labels_list, axis=0)

        return data_list, labels_list

    def __getitem__(self, index):
        im, label = self.data[index], self.labels[index]
        label = self.one_hot_map[label]

        return im, label


class Places(Dataset):
    def __init__(self, train=True, augment=False):
        self.places_path = places_root

        if train:
            base_path = osp.join(self.places_path, 'train')
        else:
            base_path = osp.join(self.places_path, 'val')

        files = os.listdir(base_path)

        self.images = []
        self.labels = []
        self.scenes = []
        counter = 0

        for f in files:
            folder_path = osp.join(base_path, f)
            images = os.listdir(folder_path)
            self.images.extend([osp.join(folder_path, image) for image in images])
            self.labels.extend([counter for _ in images])
            self.scenes.append(f)
            counter = counter + 1

        self.one_hot_map = np.eye(max(self.labels)+1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im, label = self.images[index], self.labels[index]
        im = imread(im)
        im = imresize(im, (256, 256))

        if len(im.shape) == 2:
            im = np.tile(im[:, :, None], (1, 1, 3))
        else:
            im = im[:, :, :3]

        return im, label


class PlacesRoom(Dataset):
    def __init__(self, train=True, augment=False):
        self.places_path = places_root

        if train:
            base_path = osp.join(self.places_path, 'train')
        else:
            base_path = osp.join(self.places_path, 'val')

        files = os.listdir(base_path)

        self.images = []
        self.labels = []
        self.scenes = []
        counter = 0
        # room_scenes = ["alcove", "attic", "ballroom", "basement", "bathroom", "bedchamber", "bedroom", "classroom"
        #                "bow_window-indoor", "childs_room", "closet", "dining_room", "dorm_room",
        #                "garage-indoor", "home_office", "home_theater", "hotel_room", "kitchen",
        #                "laundromat", "living_room", "nursery", "pantry", "playroom", "recreation_room",
        #                "shower", "staircase", "television_room", "utility_room", "wet_bar"]
        room_scenes = ["classroom", "mansion", "patio", "airport_terminal", "beauty_salon", "closet", "dorm_room", "home_office", "bedroom", "engine_room", "hospital_room", "martial_arts_gym", "shed", "cockpit", "hotel_outdoor", "apartment_building_outdoor", "bookstore", "coffee_shop", "hotel_room", "shopfront", "conference_center", "shower", "conference_room", "motel", "pulpit", "fire_escape", "art_gallery", "art_studio", "corridor", "museum_indoor", "railroad_track", "inn_outdoor", "music_studio", "attic", "nursery", "auditorium", "residential_neighborhood", "cafeteria", "office", "restaurant", "waiting_room", "office_building", "restaurant_kitchen", "stage_indoor", "ballroom", "game_room", "kitchen", "restaurant_patio", "staircase", "banquet_hall", "bar", "dinette_home", "living_room", "swimming_pool_outdoor", "basement", "dining_room", "lobby", "parlor", "locker_room"]
        print("length of room scenes ", len(room_scenes))


        for f in files:
            if f in room_scenes:
                folder_path = osp.join(base_path, f)
                images = os.listdir(folder_path)
                self.images.extend([osp.join(folder_path, image) for image in images])
                self.labels.extend([counter for _ in images])
                self.scenes.append(f)
                counter = counter + 1

        self.one_hot_map = np.eye(max(self.labels)+1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im, label = self.images[index], self.labels[index]
        im = imread(im)
        im = imresize(im, (256, 256))

        if len(im.shape) == 2:
            im = np.tile(im[:, :, None], (1, 1, 3))
        else:
            im = im[:, :, :3]

        return im, label


class PlacesOutdoor(Dataset):
    def __init__(self, train=True, augment=False):
        self.places_path = places_root

        if train:
            base_path = osp.join(self.places_path, 'train')
        else:
            base_path = osp.join(self.places_path, 'val')

        files = os.listdir(base_path)

        self.images = []
        self.labels = []
        self.scenes = []
        counter = 0
        room_scenes = ['abbey', 'alley', 'amphitheater', 'amusement_park', 'aqueduct', 'arch', 'apartment_building_outdoor', 'badlands', 'bamboo_forest', 'baseball_field', 'basilica', 'bayou', 'boardwalk', 'boat_deck', 'botanical_garden', 'bridge', 'building_facade', 'butte', 'campsite', 'canyon', 'castle', 'cemetery', 'chalet', 'coast', 'construction_site', 'corn_field', 'cottage_garden', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'cathedral_outdoor', 'church_outdoor', 'dam', 'dock', 'driveway', 'desert_sand', 'desert_vegetation', 'doorway_outdoor', 'excavation', 'fairway', 'fire_escape', 'fire_station', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'field_cultivated', 'field_wild', 'garbage_dump', 'gas_station', 'golf_course', 'harbor', 'herb_garden', 'highway', 'hospital', 'hot_spring', 'hotel_outdoor', 'iceberg', 'igloo', 'islet', 'ice_skating_rink_outdoor', 'inn_outdoor', 'kasbah', 'lighthouse', 'mansion', 'marsh', 'mausoleum', 'medina', 'motel', 'mountain', 'mountain_snowy', 'market_outdoor', 'monastery_outdoor', 'ocean', 'office_building', 'orchard', 'pagoda', 'palace', 'parking_lot', 'pasture', 'patio', 'pavilion', 'phone_booth', 'picnic_area', 'playground', 'plaza', 'pond', 'racecourse', 'raft', 'railroad_track', 'rainforest', 'residential_neighborhood', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'rope_bridge', 'ruin', 'runway', 'sandbar', 'schoolhouse', 'sea_cliff', 'shed', 'shopfront', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'swamp', 'stadium_baseball', 'stadium_football', 'swimming_pool_outdoor', 'television_studio',1, 'topiary_garden', 'tower', 'train_railway', 'tree_farm', 'trench', 'temple_east_asia', 'temple_south_asia', 'track_outdoor', 'underwater_coral_reef', 'valley', 'vegetable_garden', 'veranda', 'viaduct', 'volcano', 'waiting_room',1, 'water_tower', 'watering_hole', 'wheat_field', 'wind_farm', 'windmill', 'yard']


        for f in files:
            if f in room_scenes:
                folder_path = osp.join(base_path, f)
                images = os.listdir(folder_path)
                self.images.extend([osp.join(folder_path, image) for image in images])
                self.labels.extend([counter for _ in images])
                self.scenes.append(f)
                counter = counter + 1

        self.one_hot_map = np.eye(max(self.labels)+1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im, label = self.images[index], self.labels[index]
        im = imread(im)
        im = imresize(im, (256, 256))

        if len(im.shape) == 2:
            im = np.tile(im[:, :, None], (1, 1, 3))
        else:
            im = im[:, :, :3]

        return im, label


if __name__ == "__main__":
    from imageio import imwrite
    # dataset = Places()
    dataset = PlacesRoom()
    print(len(dataset.room_scenes))
    im, label = dataset[1]
    import pdb
    pdb.set_trace()
    print(label)
    print(label.shape)
    imwrite("test.png", im)
    import pdb
