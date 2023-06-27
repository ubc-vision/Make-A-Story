import os, re
import numpy as np
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json

from mugen_data.coinrun.game import Game
from mugen_data.coinrun.construct_from_json import define_semantic_color_map, generate_asset_paths, load_assets, load_bg_asset, draw_game_frame
from mugen_data.video_utils import label_color_map
from jukebox.utils.io import load_audio
from mugen_data.models.audio_vqvae.hparams import AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_LENGTH
import random


class StoryImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, im_input_size,
                 out_img_folder = None,
                 mode='train',
                 video_len = 5, fid = False):

        self.max_label = 21
        self.resolution = im_input_size
        self.sample_every_n_frames = 1
        self.sequence_length = 1
        self.fixed_start_idx = True
        self.train=False
        random.seed(30)
        dataset_json_file = os.path.join(img_folder, f"{mode}.json")
        with open(dataset_json_file, "r") as f:
            all_data = json.load(f)
        all_data["metadata"]["data_folder"] =  all_data["metadata"]["data_folder"].replace('/checkpoint/thayes427','/ubc/cs/research/shield/datasets/coinrun')
        self.dataset_metadata = all_data["metadata"]

        self.data = []

        count = 0 
        for data_sample in all_data["data"]:

            bg_list = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
            alien_name_list = ['Tony', 'Lisa', 'Jhon']
            alien_name = random.sample(alien_name_list,1)[0]      # Select random Mugen
            world_theme_n = random.randint(0, 5)     # select random background
            
            data_sample['video']['world_theme_n'] = world_theme_n
            data_sample['video']['alien_name'] = alien_name

            if data_sample["video"]["num_frames"] > (self.sequence_length - 1) * self.sample_every_n_frames:
                self.data.append(data_sample)
            
            #if count == 200:
                #break;
            count = count + 1
        
        print(f"NUMBER OF FILES LOADED: {len(self.data)}")
        self.init_game_assets()

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(im_input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 
        else:
            self.transform = transforms.Compose([ 
                transforms.ToPILImage(),   
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),  
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 

        if fid:
            #self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_input_size, im_input_size)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_input_size, im_input_size)),transforms.ToTensor()])
        #import ipdb;ipdb.set_trace()
        #self.modified_loader(0)

    def modified_loader(self, idx):
        #dpath = "MugenClipDescriptions/" + str(idx)
        #os.mkdir(dpath)
        number_clip = 0
        alien_name_list = ['Tony', 'Lisa', 'Jhon']
        bg_list = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
        change_bg = random.randint(0, number_clip)
        self.load_json_file(idx)
        self.game.world_theme_n = self.data[idx]['video']['world_theme_n'] #random.randint(0, 5)     # select random background
        start_idx, end_idx = self.get_start_end_idx()
        alien_name = self.data[idx]['video']['alien_name'] #random.sample(alien_name_list,1)[0]      # Select random Mugen
        game_video = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
        rand_idx = torch.randint(low=1, high=len(self.data[idx]["annotations"]), size=(1,)).item() if self.train else 1
        text_desc = self.data[idx]["annotations"][rand_idx]["text"].replace('Mugen', alien_name)
        init_bg = self.game.world_theme_n

        if text_desc.endswith('.'):
            text_desc = text_desc[:-1] + ' in ' + bg_list[self.game.world_theme_n]
        else:
            text_desc = text_desc + ' in ' + bg_list[self.game.world_theme_n]

        if alien_name == "Lisa":
            replace_list = ['Meanwhile ', 'On the other hand her friend ', 'By this time her friend ', 'In the meantime ', 'The time between her friend ']
        else:
            replace_list = ['Meanwhile ', 'On the other hand his friend ', 'By this time his friend ', 'In the meantime ', 'The time between his friend ']
        character1 = alien_name
        for i in range(number_clip):
            if not text_desc.endswith('.'):
                text_desc = text_desc + "; "
            else:
                #text_desc = text_desc.replace('.',';')
                last_occur_idx = text_desc.rfind('.')
                last_occur_idx2 = last_occur_idx + len(text_desc) - 1
                text_desc = text_desc[:last_occur_idx] + ';' + text_desc[last_occur_idx2:]
            idx = random.randint(0, len(self.data)-1)
            self.load_json_file(idx)
            if i == change_bg :
                while True:
                    number = random.randint(0, 5)
                    if self.game.world_theme_n != number:
                        self.game.world_theme_n = number
                        break
                while True:
                    alien_name2 = random.sample(alien_name_list,1)[0]
                    if alien_name != alien_name2:
                        alien_name = alien_name2
                        break
            else:
                self.game.world_theme_n = init_bg

            start_idx, end_idx = self.get_start_end_idx()
            game_video1 = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
            game_video = torch.cat([game_video,game_video1],0)
            rand_idx = torch.randint(low=1, high=len(self.data[idx]["annotations"]), size=(1,)).item() if self.train else 1
            text_desc1 = self.data[idx]["annotations"][rand_idx]["text"]

            if not i == change_bg:
                if i == change_bg+1 :
                    text_desc1 = text_desc1.replace('Mugen', character1)
                else:
                    if alien_name == "Lisa":
                        text_desc1 = text_desc1.replace('Mugen', "She")
                    else:
                        text_desc1 = text_desc1.replace('Mugen', "He")
            else:
                text_desc1 = text_desc1.replace('Mugen', alien_name)
                text_desc1 = random.choice(replace_list) + text_desc1
                if text_desc1.endswith('.'):
                    text_desc1 = text_desc1[:-1] + ' in ' + bg_list[self.game.world_theme_n]
                else:
                    text_desc1 = text_desc1 + ' in '+ bg_list[self.game.world_theme_n]

            text_desc = text_desc + text_desc1

        '''with open('MugenClipDescriptions.txt', 'a') as f:
            f.write(text_desc + '\n\n' )

        for i in range(len(game_video)):
            PIL.Image.fromarray(game_video[i].numpy()).save(dpath+'/%s.png' % i)'''
        
        game_video_final = torch.zeros(game_video.shape[0], game_video.shape[3], game_video.shape[2], game_video.shape[1])
        for i in range(game_video.shape[0]):
            game_video_final[i] = self.transform(game_video[i].permute(2,0,1))

        game_video_final_ldm = game_video/127.5 - 1.0
        
        label = torch.zeros(len(alien_name_list))
        bg_label = torch.zeros(len(bg_list))

        for i in range(len(bg_list)):
            if i == init_bg:
                bg_label[i] = 1

        for i in range(len(alien_name_list)):
            if alien_name_list[i] == alien_name:
                label[i] = 1
        
        return game_video_final, game_video_final_ldm, label, bg_label, text_desc


    # initialize game assets
    def init_game_assets(self):
        self.game = Game()
        self.game.load_json(os.path.join(self.dataset_metadata["data_folder"], self.data[0]["video"]["json_file"]))
        # NOTE: only supports rendering square-size coinrun frame for now
        self.game.video_res = self.resolution

        semantic_color_map = define_semantic_color_map(self.max_label)

        # grid size for Mugen/monsters/ground
        self.kx: float = self.game.zoom * self.game.video_res / self.game.maze_w
        self.ky: float = self.kx

        # grid size for background
        zx = self.game.video_res * self.game.zoom
        zy = zx

        # NOTE: This is a hacky solution to switch between theme assets
        # Sightly inefficient due to Mugen/monsters being loaded twice
        # but that only a minor delay during init
        # This should be revisited in future when we have more background/ground themes

        self.game.background_themes.append("backgrounds/background-2/Background_2.png")
        self.game.background_themes.append("backgrounds/background-2/airadventurelevel1.png")
        self.game.background_themes.append("backgrounds/background-2/airadventurelevel2.png")
        self.game.background_themes.append("backgrounds/background-2/airadventurelevel3.png")
        self.game.ground_themes.append("Grass")
        self.game.ground_themes.append("Dirt")
        self.game.ground_themes.append("Stone")
        self.game.ground_themes.append("Sand")

        self.total_world_themes = len(self.game.background_themes)
        self.asset_map = {}

        for world_theme_n in range(self.total_world_themes):
            # reset the paths for background and ground assets based on theme
            self.game.world_theme_n = world_theme_n
            asset_files = generate_asset_paths(self.game)

            # TODO: is it worth to load assets separately for game frame and label?
            # this way game frame will has smoother character boundary
            self.asset_map[world_theme_n] = load_assets(
                asset_files, semantic_color_map, self.kx, self.ky, gen_original=False
            )

            # background asset is loaded separately due to not following the grid
            self.asset_map[world_theme_n]['background'] = load_bg_asset(
                asset_files, semantic_color_map, zx, zy
            )


    def __len__(self):
        return len(self.data)

    def load_json_file(self, idx):
        self.game.load_json(os.path.join(self.dataset_metadata["data_folder"], self.data[idx]["video"]["json_file"]))
        self.game.video_res = self.resolution

    def get_start_end_idx(self, valid_frames=None):
        start_idx = 0
        end_idx = len(self.game.frames)
        if self.sequence_length is not None:
            assert (self.sequence_length - 1) * self.sample_every_n_frames < end_idx, \
                f"not enough frames to sample {self.args.sequence_length} frames at every {self.args.sample_every_n_frames} frame"
            if self.fixed_start_idx:
                start_idx = 0
            else:
                if valid_frames:
                    # we are sampling frames from a full json and we need to ensure that the desired
                    # class is in the frame range we sample. Resample until this is true
                    resample = True
                    while resample:
                        start_idx = torch.randint(
                            low=0,
                            high=end_idx - (self.sequence_length - 1) * self.sample_every_n_frames,
                            size=(1,)
                        ).item()
                        for valid_frame_range in valid_frames:
                            if isinstance(valid_frame_range, list):
                                # character ranges
                                st_valid, end_valid = valid_frame_range
                            else:
                                # game event has a single timestamp
                                st_valid, end_valid = valid_frame_range, valid_frame_range
                            if end_valid >= start_idx and start_idx + self.sequence_length * self.sample_every_n_frames >= st_valid:
                                # desired class is in the sampled frame range, so stop sampling
                                resample = False
                else:
                    start_idx = torch.randint(
                        low=0,
                        high=end_idx - (self.sequence_length - 1) * self.sample_every_n_frames,
                        size=(1,)
                    ).item()
            end_idx = start_idx + self.sequence_length * self.sample_every_n_frames
        return start_idx, end_idx

    def get_game_video(self, start_idx, end_idx, alien_name='Mugen'):
        frames = []
        for i in range(start_idx, end_idx, self.sample_every_n_frames):
            img = draw_game_frame(
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=True, alien_name=alien_name
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        return torch.vstack(frames)


    def get_game_audio(self, wav_filename):
        data, _ = load_audio(wav_filename, sr=self.args.audio_sample_rate, offset=0, duration=self.args.audio_sample_length)
        data = torch.as_tensor(data).permute(1, 0)
        return data

    def get_smap_video(self, start_idx, end_idx, alien_name='Mugen'):
        frames = []
        for i in range(start_idx, end_idx, self.args.sample_every_n_frames):
            img = draw_game_frame(
                self.game, i, self.asset_map[self.game.world_theme_n], self.kx, self.ky, gen_original=False,
                bbox_smap_for_agent=self.bbox_smap_for_agent, bbox_smap_for_monsters=self.bbox_smap_for_monsters, alien_name=alien_name
            )
            frames.append(torch.unsqueeze(torch.as_tensor(np.array(img)), dim=0))
        # typical output shape is 16 x 256 x 256 x 1 (sequence_length=16, resolution=256)
        return torch.unsqueeze(torch.vstack(frames), dim=3)


    def __getitem__(self, idx):

        return self.modified_loader(idx)
