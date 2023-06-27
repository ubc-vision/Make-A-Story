import os, pickle, re, csv
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from random import randrange
from collections import Counter
import json
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


all_characters = ["Pororo", "Loopy", "Crong", "Eddy", "Poby", "Petty", "Tongtong", "Rody", "Harry", "pororo", "loopy", "crong", "eddy", "poby", "petty", "tongtong", "rody", "harry"]
female = ["Petty", "Loopy", "petty", "loopy"]


class VideoFolderDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.lengths = []
        self.followings = []
        #self.dir_path = data_folder
        self.total_frames = 0
        self.images = []
        self.dir_path = None
        self.descriptions = None
        self.ids = None
        self.labels = None
        self.video_len = 0


    def story_dataset(self, data_folder, cache=None, min_len=3, mode='train'):
        self.dir_path = cache
        self.out_img_folder = ""
        im_input_size = 256
        self.video_len = min_len 
        self.labels = np.load(os.path.join(self.dir_path, 'labels.npy'), allow_pickle=True, encoding='latin1').item() 
        self.descriptions_original = np.load(os.path.join(self.dir_path, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()

        dataset = ImageFolder(data_folder)
        counter = np.load(os.path.join(data_folder, 'frames_counter.npy'), allow_pickle=True).item()
         
        self.labels = np.load(os.path.join(data_folder, 'labels.npy'), allow_pickle=True, encoding='latin1').item()
        
        if cache is not None and os.path.exists(os.path.join(cache, 'img_cache' + str(min_len) + '.npy')) and os.path.exists(os.path.join(cache,'following_cache' + str(min_len) +  '.npy')):
            self.images = np.load(os.path.join(cache,'img_cache' + str(min_len) + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(cache,'following_cache' + str(min_len) + '.npy'))
        else: 
            for idx, (im, _) in enumerate(tqdm(dataset, desc="Counting total number of frames")):
                img_path, _ = dataset.imgs[idx]
                v_name = img_path.replace(data_folder,'')
                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))
                v_name = re.sub(r"[0-9]+.png",'', v_name)
                if id > counter[v_name[1:]] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(data_folder, ''))
                self.followings.append(following_imgs)
            np.save(os.path.join(data_folder, 'img_cache' + str(min_len) + '.npy'), self.images)
            np.save(os.path.join(data_folder, 'following_cache' + str(min_len) + '.npy'), self.followings)

        train_id, val_id, test_id = np.load(os.path.join(self.dir_path, "train_seen_unseen_ids.npy"), allow_pickle=True)
       
        if mode == 'train':
            self.ids = np.sort(train_id)
        elif mode =='val':
            self.ids = np.sort(val_id)
        elif mode =='test':
            self.ids = np.sort(test_id)
        else:
            raise ValueError
         
        if mode == "train": 
            self.transform = transforms.Compose([transforms.Resize(im_input_size),
         			transforms.ToTensor(),
         			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ])
        else:
            self.transform = transforms.Compose([transforms.Resize(im_input_size),
         			transforms.ToTensor()])
        self.count = 0
        '''for i in range(len(self.ids)):
            example = self.getSampleItem(i)
            dpath = os.path.join("prororo_image", str(i))
            os.mkdir(dpath)
            with open('prororo.txt', 'a') as f:
                f.write(example['caption'] + '\n\n' )
            for j in range(len(example['image'])):
                PIL.Image.fromarray((((example['image'][j].numpy()+1)/2)*255).astype(np.uint8)).save(dpath+'/%s.png' % j)
        import ipdb;ipdb.set_trace()'''

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def getSampleItem(self, item):
        # single image input
        src_img_id = self.ids[item]
        
        all_img_ids = [str(self.images[src_img_id])[1:-4]]
        tgt_img_ids = [str(self.followings[src_img_id][i])[1:-4] for i in range(self.video_len)]
        all_img_ids = all_img_ids + tgt_img_ids

        for idx, img_id in enumerate(all_img_ids):
            src_img_path = os.path.join(self.dir_path, img_id+".png")
            src_image = self.transform(self.sample_image(PIL.Image.open(src_img_path).convert('RGB')))
      
            if idx == 0:
                images = src_image.unsqueeze(0)
                text = self.descriptions_original[img_id][0] + ";"
               
                char_name = [x for x in all_characters if x in text]
                if len(char_name) > 1:
                    if len(text[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in text[:-1].split(".")[-1]]
                        if len(char_name)>0:
                            imidiate_char = char_name[0]
                        else:
                            imidiate_char = ""
                    else:
                        imidiate_char = char_name[0]
                elif len(char_name) == 1:
                    imidiate_char = char_name[0]
                else:
                    imidiate_char = ""
            else:
                images = torch.cat([images,src_image.unsqueeze(0)],0)
                text1 = self.descriptions_original[img_id][0]
                char_name = [x for x in all_characters if x in text1]
                if len(char_name) > 1:
                    if len(text1[:-1].split('.')) > 1:
                        char_name = [x for x in all_characters if x in text1[:-1].split(".")[-1]]
                        if len(char_name)>0:
                            char_name = char_name[0]
                        else:
                            char_name = ""
                    else:
                        char_name = char_name[0]
                elif len(char_name) == 1:
                    char_name = char_name[0]
                else:
                    char_name = ""
                if char_name != "" and char_name == imidiate_char:
                    if char_name in female:
                        replace_char = "She"
                    else:
                        replace_char = "He" 
                    text1 = text1.replace(char_name, replace_char)
                else:
                    imidiate_char = char_name
                text = text + text1 + " ;" 

        #labels = [torch.tensor(self.labels[img_id]) for img_id in all_img_ids]
        example = dict()
        #example['char'] = torch.stack(labels)
        #example['background'] = [self.settings[globalID] for globalID in globalIDs]
        example['image'] = images.permute(0,2,3,1)
        example['caption'] = text[:-1]
        #ref = ["He", "She", "he","she" ]
        #self.count = self.count + sum(i in text.lower() for i in ref)
        #print(self.count)
     
        return example

    def __getitem__(self, item):
        return self.getSampleItem(item)

    def __len__(self):
        return len(self.ids)


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


