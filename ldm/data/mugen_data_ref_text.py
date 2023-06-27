import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import PIL
#from taming.data.mugen_base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import json
import torch
from mugen_data.coinrun.game import Game
from mugen_data.coinrun.construct_from_json import define_semantic_color_map, generate_asset_paths, load_assets, load_bg_asset, draw_game_frame
from mugen_data.video_utils import label_color_map
from jukebox.utils.io import load_audio
from mugen_data.models.audio_vqvae.hparams import AUDIO_SAMPLE_RATE, AUDIO_SAMPLE_LENGTH
import random


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def mugen_dataset(self, json_file, get_game_frame, get_text_desc, get_audio, split, sample_every_n_frames, sequence_length, data_path, use_manual_annotation, resolution):
        self.max_label = 21
        self.resolution = resolution
        self.get_audio = get_audio
        self.get_game_frame = get_game_frame
        self.get_text_desc = get_text_desc
        self.use_manual_annotation = use_manual_annotation
        self.sample_every_n_frames = sample_every_n_frames
        self.sequence_length = sequence_length
        self.fixed_start_idx = True
        self.train = False

        if split == 'train':
            self.train = True

        assert get_game_frame or get_audio or get_text_desc, \
                "Need to return at least one of game frame, audio, or text desc"
        dataset_json_file = json_file
        print(f"LOADING FROM JSON FROM {dataset_json_file}...")
        with open(dataset_json_file, "r") as f:
            all_data = json.load(f)

        all_data["metadata"]["data_folder"] =  all_data["metadata"]["data_folder"].replace('/checkpoint/thayes427','/ubc/cs/research/shield/datasets/coinrun')
        self.dataset_metadata = all_data["metadata"]
        self.data = []
        i = 0         
        #import ipdb;ipdb.set_trace()
        for data_sample in all_data["data"]:
            '''bg_list = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
            alien_name_list = ['Tony', 'Lisa', 'Jhon']
            #alien_name_list = ['Mugen', 'LadyMugen', 'BabyMugen']
            alien_name = random.sample(alien_name_list,1)[0]      # Select random Mugen
            world_theme_n = random.randint(0, 5)     # select random background
            data_sample['video']['world_theme_n'] = world_theme_n
            data_sample['video']['alien_name'] = alien_name'''
            if data_sample["video"]["num_frames"] > (sequence_length - 1) * sample_every_n_frames:
                self.data.append(data_sample)
                i = i+1
            #if i > 2000:
                #break
                
        print(f"NUMBER OF FILES LOADED: {len(self.data)}")
        self.init_game_assets()

        ###############################################################################################################
        '''import ipdb; ipdb.set_trace()
        for i in range(100):
            images, text = self.modified_loader(i)
        import ipdb;ipdb.set_trace()'''


    def modified_loader(self, idx):
        #dpath = "MugenClipDescriptions/" + str(idx)
        #os.mkdir(dpath)

        number_clip = 3
        no_of_bg = 5
        alien_name_list = ['Tony', 'Lisa', 'Jhon']
        #alien_name_list = ['Mugen', 'LadyMugen', 'BabyMugen']
        bg_list = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
        change_bg = random.randint(0, number_clip)
        self.load_json_file(idx)
        self.game.world_theme_n = random.randint(0, no_of_bg)     # select random background
        start_idx, end_idx = self.get_start_end_idx()
        alien_name = random.sample(alien_name_list,1)[0]      # Select random Mugen
        game_video = self.get_game_video(start_idx, end_idx, alien_name=alien_name)
        rand_idx = torch.randint(low=1, high=len(self.data[idx]["annotations"]), size=(1,)).item() if self.train else 1
        text_desc = self.data[idx]["annotations"][rand_idx]["text"].replace('Mugen', alien_name)
        init_bg = self.game.world_theme_n
        
        if text_desc.endswith('.'):
            text_desc = text_desc[:-1] + ' in ' + bg_list[self.game.world_theme_n]
        else:
            text_desc = text_desc + ' in ' + bg_list[self.game.world_theme_n]

        #text_desc = alien_name + ' in ' + bg_list[self.game.world_theme_n]

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
            '''if i == change_bg :
                while True:
                    number = random.randint(0, no_of_bg)
                    if self.game.world_theme_n != number: 
                        self.game.world_theme_n = number
                        break
                while True:
                    alien_name2 = random.sample(alien_name_list,1)[0]
                    if alien_name != alien_name2:
                        alien_name = alien_name2
                        break
            else:
                self.game.world_theme_n = init_bg'''
            self.game.world_theme_n = init_bg
            start_idx, end_idx = self.get_start_end_idx()

            game_video1 = self.get_game_video(start_idx, end_idx, alien_name=alien_name)    
            game_video = torch.cat([game_video,game_video1],0)
            rand_idx = torch.randint(low=1, high=len(self.data[idx]["annotations"]), size=(1,)).item() if self.train else 1
            text_desc1 = self.data[idx]["annotations"][rand_idx]["text"]
            
            '''if not i == change_bg:
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
                    text_desc1 = text_desc1 + ' in '+ bg_list[self.game.world_theme_n]'''

            if alien_name == 'Lisa':
                text_desc1 = text_desc1.replace('Mugen', "She")
            else:
                text_desc1 = text_desc1.replace('Mugen', "He")

            text_desc = text_desc + text_desc1
            
        '''with open('MugenClipDescriptions.txt', 'a') as f:
            f.write(text_desc + '\n\n' )

        for i in range(len(game_video)):
            PIL.Image.fromarray(game_video[i].numpy()).save(dpath+'/%s.png' % i)'''

        return game_video.squeeze(0), text_desc
                
                           
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
        
        #del self.game.background_themes[-1]  #######
       
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

        result_dict = {}
        result_dict['video'], result_dict['text'] = self.modified_loader(idx)
        example = dict()
        example["image"] = result_dict['video']/127.5 - 1.0 #self.data[i]
        example["caption"] = result_dict['text']

        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_json, get_game_frame, get_text_desc, get_audio, split, sample_every_n_frames, sequence_length, data_path, use_manual_annotation, resolution):
        super().__init__()
        
        CustomBase.mugen_dataset(self, training_json, get_game_frame, get_text_desc, get_audio, split, sample_every_n_frames, sequence_length, data_path, use_manual_annotation, resolution)


class CustomTest(CustomBase):
    def __init__(self, size, test_json, get_game_frame, get_text_desc, get_audio, split, sample_every_n_frames, sequence_length, data_path, use_manual_annotation, resolution):
        super().__init__()
      
        CustomBase.mugen_dataset(self, test_json, get_game_frame, get_text_desc, get_audio, split, sample_every_n_frames, sequence_length, data_path, use_manual_annotation, resolution)


