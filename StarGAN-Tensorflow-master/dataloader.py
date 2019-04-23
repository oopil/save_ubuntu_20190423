from class_metabot import MetaBot
import imageio

class DataLoader():
    def __init__(self):
        self.frd = 0
        self.data_tr = [] # train data must have both mri and pet images
        # data with only pet or mri
        self.data_tst_mri = []
        self.data_tst_pet = []

        self.label_tr = []
        self.label_tst_mri = []
        self.label_tr_pet = []
        self.label_tst = [] # we do not need this
        self.meta_info = []

    def make_data_set(self):


    def set_meta_info(self, meta_l):
        self.meta_info = meta_l


class DataInfo():
    def __init__(self):
        self.meta = []
        self.format = 'hdr'
        # self.format = 'img'
        self.grp = ['AD', 'MCI', 'NORMAL']

    def rd_fld(self, dir):
        bot = MetaBot('')
        fld_l = bot.get_file_list(dir)
        print(fld_l)
        for fld in fld_l:
            path = bot.join_path(dir,fld)
            if bot.is_dir(path):
                self.rd_person(path)
        del bot

    def rd_person(self, dir):
        bot = MetaBot('')
        file_l = bot.get_file_list(dir)
        # print(file_l)
        datum = Datum()
        tmp_path = []
        for file in file_l:
            file_path = bot.join_path(dir, file)
            if self.is_format(file):
                tmp_path.append(file_path)
        datum.set_info(tmp_path, self.get_gr_num(dir))
        datum.print()
        self.meta.append(datum)
        del bot

    def is_format(self, name):
        if self.format in name: return True
        return False

    def get_gr_num(self, path):
        index = 0
        for gr in self.grp:
            if gr in path:
                return index
            index = index+1
        print('cannot find group number in get_gr_num.')

    def get_meta_info(self):
        return self.meta

class Datum():
    def __init__(self):
        self.name = 'None'
        self.gr_num = 0
        self.mri_is = False
        self.pet_is = False
        self.mri_path = 'None'
        self.pet_path = 'None'

    def set_info(self, path_l, gr_num):
        self.gr_num = gr_num
        for path in path_l:
            path_split = path.split('/')
            # print(path_split)
            self.name = path_split[-2]
            if self.is_mri(path):
                self.mri_is = True
                self.mri_path = path
            if self.is_pet(path):
                self.pet_is = True
                self.pet_path = path

    def is_pet(self,name):
        if 'pet' in name:
            return True
        return False

    def is_mri(self, name):
        if 'strip' in name:
            return True
        return False

    def print(self):
        info = []
        info.append(self.name)
        info.append(self.gr_num)
        info.append(self.mri_is)
        info.append(self.mri_path)
        info.append(self.pet_is)
        info.append(self.pet_path)
        print(info)

    def has_mri(self):
        if self.mri_is: return True
        return False

    def has_pet(self):
        if self.pet_is: return True
        return False

    def get_pet(self):
        return self.pet_path

    def get_mri(self):
        return self.mri_path

    def get_gr(self):
        return self.gr_num

def load_data():
    bot = MetaBot('')
    base_fld = '/home/sp/Datasets/MRI_PETDATA/MRI_PETDATA_0927'
    class_name = ['linearToJakobAD','linearToJakobMCI','linearToJakobNORMAL']
    loader = DataInfo()
    for name in class_name:
        loader.rd_fld(bot.join_path(base_fld,name))

load_data()
