import os, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from random import randrange
from collections import Counter
import json
import torchvision.transforms as transforms

unique_characters = ["Wilma", "Fred", "Betty", "Barney", "Dino", "Pebbles", "Mr Slate"]
female = ["Wilma", "Betty", "Pebbles"]


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.lengths = []
        self.followings = {}
        #self.dir_path = data_folder
        self.total_frames = 0

    def story_dataset(self, data_folder, cache=None, min_len=3, mode='train'):
        self.dir_path = data_folder
        self.out_img_folder = ""
        im_input_size = 256 
        # train_id, test_id = np.load(self.dir_path + 'train_test_ids.npy', allow_pickle=True, encoding='latin1')
        splits = json.load(open(os.path.join(self.dir_path, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]

        if os.path.exists(cache + 'following_cache' + str(min_len) +  '.npy'):
            self.followings = pickle.load(open(cache + 'following_cache' + str(min_len) + '.pkl', 'rb'))
        else:
            all_clips = train_id + val_id + test_id
            all_clips.sort()
            for idx, clip in enumerate(tqdm(all_clips, desc="Counting total number of frames")):
                season, episode = int(clip.split('_')[1]), int(clip.split('_')[3])
                has_frames = True
                for c in all_clips[idx+1:idx+min_len+1]:
                    s_c, e_c = int(c.split('_')[1]), int(c.split('_')[3])
                    if s_c != season or e_c != episode:
                        has_frames = False
                        break
                if has_frames:
                    self.followings[clip] = all_clips[idx+1:idx+min_len+1]
                else:
                    continue
            pickle.dump(self.followings, open(os.path.join(data_folder, 'following_cache' + str(min_len) + '.pkl'), 'wb'))

        ### character
        if os.path.exists(os.path.join(data_folder, 'labels.pkl')):
            self.labels = pickle.load(open(os.path.join(data_folder, 'labels.pkl'), 'rb'))
        else:
            print("Computing and saving labels")
            annotations = json.load(open(os.path.join(data_folder, 'flintstones_annotations_v1-0.json'), 'r'))
            self.labels = {}
            for sample in annotations:
                sample_characters = [c["entityLabel"].strip().lower() for c in sample["characters"]]
                self.labels[sample["globalID"]] = [1 if c.lower() in sample_characters else 0 for c in unique_characters]
            pickle.dump(self.labels, open(os.path.join(data_folder, 'labels.pkl'), 'wb'))

        ### description and backgorund (settings)
        self.descriptions = {}
        self.settings = {}
        self.all_settings = []
        self.characters = {}
        annotations = json.load(open(os.path.join(data_folder, 'flintstones_annotations_v1-0.json'), 'r'))
        for sample in annotations:
            self.descriptions[sample["globalID"]] = sample["description"]
            self.settings[sample["globalID"]] = sample["setting"]
            self.characters[sample["globalID"]] = [c["entityLabel"].strip() for c in sample["characters"]]
            if not sample["setting"] in self.all_settings:
                self.all_settings.append(sample["setting"])

        #self.embeds = np.load(os.path.join(self.dir_path, "flintstones_use_embeddings.npy"))
        #self.sent2idx = pickle.load(open(os.path.join(self.dir_path, 'flintstones_use_embed_idxs.pkl'), 'rb'))

        self.filtered_followings = {}
        for i, f in self.followings.items():
            #print(f)
            if len(f) == min_len:
                self.filtered_followings[i] = f
            else:
                continue
        self.followings = self.filtered_followings

        train_id = [tid for tid in train_id if tid in self.followings]
        val_id = [vid for vid in val_id if vid in self.followings]
        test_id = [tid for tid in test_id if tid in self.followings]
        
        if mode == 'train':
            self.orders = train_id
            self.transform = transforms.Compose([
                    #transforms.RandomResizedCrop(im_input_size),
                    #transforms.RandomHorizontalFlip(),
                    transforms.Resize(im_input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif mode =='val':
            self.orders = val_id
            self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    #transforms.CenterCrop(im_input_size),
                    transforms.ToTensor()
                ])
        elif mode == 'test':
            self.orders = test_id
            self.transform = transforms.Compose([
                    transforms.Resize(im_input_size),
                    #transforms.CenterCrop(im_input_size),
                    transforms.ToTensor()
                ])
        else:
            raise ValueError
        print("Total number of clips {}".format(len(self.orders)))
       
        '''for i in range(200):
            example = self.getSampleItem(i)
            #import ipdb;ipdb.set_trace()
            dpath = os.path.join("flinstone_image", str(i))
            os.mkdir(dpath)
            with open('Flinstone.txt', 'a') as f:
                f.write(example['caption'] + '\n\n' )
            for j in range(len(example['image'])):
                PIL.Image.fromarray((example['image'][j].numpy()*255).astype(np.uint8)).save(dpath+'/%s.png' % j)
        import ipdb;ipdb.set_trace()'''
 

    def getSampleItem(self, item):
        # single image input
        globalIDs = [self.orders[item]] + self.followings[self.orders[item]]
        for idx, globalID in enumerate(globalIDs):
            path = os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy')
            arr = np.load(path)
            n_frames = arr.shape[0]
            random_range = randrange(n_frames)
            im = arr[random_range]
            image = PIL.Image.fromarray(im.astype('uint8'), 'RGB')
         
            if idx == 0:
                text = self.descriptions[globalID] + " ;"
                images = self.transform(image).unsqueeze(0)
                imidiate_char = self.characters[globalID]
                #break
            else:
                text1 = self.descriptions[globalID]
                if self.characters[globalID] == imidiate_char or self.characters[globalID] in imidiate_char:
                    if len(imidiate_char) > 1:
                        replace_char = "They"
                        if len(imidiate_char) == 2:
                            char_name = imidiate_char[0].capitalize()+ " and " + imidiate_char[1].capitalize()
                            if not char_name in text1:
                                char_name = imidiate_char[1].capitalize()+ " and " + imidiate_char[0].capitalize()
                        elif len(imidiate_char) == 3:
                            char_name = imidiate_char[0].capitalize()+ ", "+ imidiate_char[1].capitalize() + " and " + imidiate_char[2].capitalize()
                    elif imidiate_char in female or self.characters[globalID] in female:
                        replace_char = "She"
                        char_name = imidiate_char[0].capitalize()
                    else:
                        replace_char = "He"
                        char_name = imidiate_char[0].capitalize()
                    text1 = text1.replace(char_name, replace_char)
                else:
                    imidiate_char = self.characters[globalID]
                text = text + text1 + " ;"
                images = torch.cat([images,self.transform(image).unsqueeze(0)],0)
         
        
        labels = [torch.tensor(self.labels[globalID]) for globalID in globalIDs]
        example = dict()
        example['char'] = labels
        example['background'] = [self.settings[globalID] for globalID in globalIDs]
        example['image'] = images.permute(0,2,3,1)
        example['caption'] = text[:-1]
     
        return example

    '''def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0, video_len, 1)[0]
        #print(se*shorter, shorter, (se+1)*shorter)
        return im.crop((0, se * shorter, shorter, (se+1)*shorter)), se'''

    def __getitem__(self, item):
        return self.getSampleItem(item)

    def __len__(self):
        return len(self.orders)


'''class StoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, return_caption=False, out_dir=None, densecap=False):
        self.dir_path = dataset.dir_path
        self.dataset = dataset
        self.transforms = transform
        self.labels = dataset.labels
        self.return_caption = return_caption

        annotations = json.load(open(os.path.join(self.dir_path, 'flintstones_annotations_v1-0.json')))
        self.descriptions = {}
        for sample in annotations:
            self.descriptions[sample["globalID"]] = sample["description"]

        if self.return_caption:
            self.init_mart_vocab()
            self.max_len = self.tokenize_descriptions()
            print("Max sequence length = %s" % self.max_len)
        else:
            self.vocab = None
        self.out_dir = out_dir

        # if densecap:
        #     self.densecap_dataset = DenseCapDataset(self.dir_path)
        # else:
        self.densecap_dataset = None

    def tokenize_descriptions(self):
        caption_lengths = []
        self.tokenized_descriptions = {}
        for img_id, descs in self.descriptions.items():
            self.tokenized_descriptions[img_id] = nltk.tokenize.word_tokenize(descs.lower())
            caption_lengths.append(len(self.tokenized_descriptions[img_id]))
        return max(caption_lengths) + 2

    def init_mart_vocab(self):

        vocab_file = os.path.join(self.dir_path, 'mart_vocab.pkl')
        if os.path.exists(vocab_file):
            vocab_from_file = True
        else:
            vocab_from_file = False

        self.vocab = Vocabulary(vocab_threshold=5,
                                vocab_file=vocab_file,
                                annotations_file=os.path.join(self.dir_path, 'flintstones_annotations_v1-0.json'),
                                vocab_from_file=vocab_from_file)

    def save_story(self, output, save_path = './'):
        all_image = []
        images = output['images_numpy']
        texts = output['text']
        for i in range(images.shape[0]):
            all_image.append(np.squeeze(images[i]))
        output = PIL.Image.fromarray(np.concatenate(all_image, axis = 0))
        output.save(save_path + 'image.png')
        fid = open(save_path + 'text.txt', 'w')
        for i in range(len(texts)):
            fid.write(texts[i] +'\n' )
        fid.close()
        return

    def _sentence_to_idx(self, sentence_tokens):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_len
        sentence_tokens = sentence_tokens[:max_t_len - 2]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.vocab.pad_word] * (max_t_len - valid_l)
        input_ids = [self.vocab.word2idx.get(t, self.vocab.word2idx[self.vocab.unk_word]) for t in sentence_tokens]

        return input_ids, mask

    def __getitem__(self, item):
        lists = self.dataset[item]
        labels = []
        images = []
        text = []
        input_ids = []
        masks= []
        sent_embeds = []
        for idx, globalID in enumerate(lists):
            if self.out_dir:
                im = PIL.Image.open(os.path.join(self.out_dir, 'img-%s-%s.png' % (item, idx))).convert('RGB')
            else:
                arr = np.load(os.path.join(self.dir_path, 'video_frames_sampled', globalID + '.npy'))
                n_frames = arr.shape[0]
                im = arr[randrange(n_frames)]
            images.append(np.expand_dims(np.array(im), axis=0))
            text.append(self.descriptions[globalID])
            labels.append(np.expand_dims(self.labels[globalID], axis = 0))
            sent_embeds.append(np.expand_dims(self.dataset.embeds[self.dataset.sent2idx[globalID]], axis = 0))

            if self.return_caption:
                input_id, mask = self._sentence_to_idx(self.tokenized_descriptions[globalID])
                input_ids.append(np.expand_dims(input_id, axis=0))
                masks.append(np.expand_dims(mask, axis=0))

        sent_embeds = np.concatenate(sent_embeds, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        images = np.concatenate(images, axis = 0)
        # image is T x H x W x C
        transformed_images = self.transforms(images)
        # After transform, image is C x T x H x W

        sent_embeds = torch.tensor(sent_embeds)
        labels = torch.tensor(np.array(labels).astype(np.float32))

        data_item = {'images': transformed_images, 'text':text, 'description': sent_embeds, 'images_numpy':images, 'labels':labels}

        if self.return_caption:
            input_ids = torch.tensor(np.concatenate(input_ids))
            masks = torch.tensor(np.concatenate(masks))
            data_item.update({'input_ids': input_ids, 'masks': masks})

        if self.densecap_dataset:
            boxes, caps, caps_len = [], [], []
            for idx, v in enumerate(lists):
                img_id = str(v).replace('.png', '')[2:-1]
                path = img_id + '.png'
                boxes.append(torch.as_tensor([ann['box'] for ann in self.densecap_dataset[path]], dtype=torch.float32))
                caps.append(torch.as_tensor([ann['cap_idx'] for ann in self.densecap_dataset[path]], dtype=torch.long))
                caps_len.append(torch.as_tensor([sum([1 for k in ann['cap_idx'] if k!= 0]) for ann in self.densecap_dataset[path]], dtype=torch.long))
            targets = {
                'boxes': torch.cat(boxes),
                'caps': torch.cat(caps),
                'caps_len': torch.cat(caps_len),
            }
            data_item.update(targets)

        return data_item

    def __len__(self):
        return len(self.dataset.orders)'''



class CustomTrain(VideoFolderDataset):
    def __init__(self, data_folder, cache=None, min_len=3, mode='train'):
        super().__init__()

        VideoFolderDataset.story_dataset(self, data_folder, cache, min_len=3, mode='train')


class CustomTest(VideoFolderDataset):
    def __init__(self, data_folder, cache=None, min_len=3, mode='test'):
        super().__init__()

        VideoFolderDataset.story_dataset(self, data_folder, cache, min_len=3, mode='test')


