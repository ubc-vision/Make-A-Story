from __future__ import print_function
from __future__ import division
import os, sys
sys.path.append('.')

import torch.nn as nn
from torchvision import models
import torch
from classifier.eval_dataloader_multi import StoryImageDataset
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

import PIL
import torchvision.utils as vutils
import argparse
from sklearn.metrics import classification_report, accuracy_score, f1_score
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from tqdm import tqdm, trange
from einops import rearrange
from PIL import Image
from torchvision import transforms
from ldm.util import instantiate_from_config
import gc

epsilon = 1e-7

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet50
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def images_to_numpy(tensor):
    generated = tensor.data.cpu().numpy().transpose(1, 2, 0)
    generated[generated < -1] = -1
    generated[generated > 1] = 1
    generated = (generated + 1) / 2 * 255
    return generated.astype('uint8')

def numpy_to_img(numpy_file, outdir, img_size):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    x = np.load(numpy_file)
    print("Numpy image file shape: ", x.shape)
    for i in tqdm(range(x.shape[0])):
        frames = x[i, :, :, :, :]
        frames = np.swapaxes(frames, 0, 1)
        # frames = torch.Tensor(frames).view(-1, 3, 64, 64)
        # frames = torch.nn.functional.upsample(frames, size=(img_size, img_size), mode="bilinear")

        # vutils.save_image(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0), 'sequence-2.png')
        all_images = images_to_numpy(vutils.make_grid(torch.Tensor(frames).view(-1, 3, 64, 64), 1, padding=0))
        # all_images = images_to_numpy(vutils.make_grid(frames, 1, padding=0))
        # print(all_images.shape)
        for j, idx in enumerate(range(64, all_images.shape[0] + 1, 64)):
            output = PIL.Image.fromarray(all_images[idx-64: idx, :, :])
            output.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            img = PIL.Image.open(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))
            if img_size != 64:
                img = img.resize((img_size, img_size,))
            img.save(os.path.join(outdir, 'img-%s-%s.png' % (i, j)))

#### evaluation on ground truth

def evaluate_gt(image_path, model_name, model_path, mode):
    # Number of classes in the dataset
    num_classes = 6
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    phase = 'eval'
    
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    if image_path.endswith('.npy'):
        numpy_to_img(image_path, image_path[:-4], input_size)
        out_img_folder = image_path[:-4]
    else:
        out_img_folder = image_path

    # Create training and validation datasets
    image_dataset = StoryImageDataset(image_path, input_size, out_img_folder=out_img_folder, mode=mode)
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 8

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    running_corrects = 0
    running_recalls = 0
    total_positives = 0

    ################ end #################
    scale = 5.0
    n_samples = 1
    n_iter = 1
    ddim_steps = 50
    ddim_eta = 0.0
    H = 256
    W = 256
    im_input_size = 299

    total_gt_positives = 0
    total_pred_positives = 0
    total_true_positives = 0
    total_frame_positives = 0
    total_char_positives = np.zeros(6)
    
    # Iterate over data.
    for i, (inputs, _, _, labels, text_decs) in tqdm(enumerate(dataloader)):
         
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        num_frame = 4
        #import ipdb;ipdb.set_trace()
        for k in range(num_frame):   #batch size
            inputs_vid = inputs[:,k].float()
            with torch.no_grad():
                outputs = model_ft(inputs_vid)
                preds = torch.round(nn.functional.sigmoid(outputs))
            all_predictions.append(preds.detach().cpu().numpy())
            labels_ = labels#[k].repeat(num_frame,1)
            all_labels.append(labels_.cpu().numpy())

            ##statistics
            iter_corrects = torch.sum(preds == labels_.float().data)
            xidxs, yidxs = torch.where(labels_.data == 1)
            iter_recalls = sum([x.item() for x in [labels_.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
            total_positives += xidxs.size(0)
            for label, pred in zip(labels_, preds):
                if torch.all(torch.eq(label.float().data, pred)):
                    story_accuracy += 1
                for l, p in zip(label, pred):
                    if torch.all(torch.eq(l.float().data, p)):
                        image_accuracy += 1
            running_corrects += iter_corrects
            running_recalls += iter_recalls

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    preds = np.round(1 / (1 + np.exp(-all_predictions)))
    char_class = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
    print(classification_report(all_labels, all_predictions, target_names=char_class))
    print("Accuracy: ", accuracy_score(all_labels, preds))

    epoch_acc = float(running_corrects) * 100 / (all_labels.shape[0] * all_labels.shape[1])
    epoch_recall = float(running_recalls) * 100 / total_positives
    #print('Manually calculated accuracy: ', epoch_acc)
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, preds), epoch_recall))


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

#### evaluation for autoencoder output during first stage
def evaluate_autoencoder(image_path, model_name, model_path, mode):
    # Number of classes in the dataset
    num_classes = 6
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    phase = 'eval'

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    if image_path.endswith('.npy'):
        numpy_to_img(image_path, image_path[:-4], input_size)
        out_img_folder = image_path[:-4]
    else:
        out_img_folder = image_path

    # Create training and validation datasets
    input_size = 256
    image_dataset = StoryImageDataset(image_path, input_size, out_img_folder=out_img_folder, mode=mode)
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 12

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    running_corrects = 0
    running_recalls = 0
    total_positives = 0

    ######## Load pretrain model #########
    '''config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, "logs/2022-08-30T13-37-48_txt2img-1p4B-train/checkpoints/epoch=000049.ckpt")'''

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-train.yaml")
    model = instantiate_from_config(config.model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device).eval()

    #sampler = DDIMSampler(model)

    ################ end #################
    scale = 5.0
    n_samples = 1
    n_iter = 1
    ddim_steps = 50
    ddim_eta = 0.0
    H = 256
    W = 256
    im_input_size = 299
    transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    # Iterate over data.
    for i, (_, inputs,_, labels, text_decs) in tqdm(enumerate(dataloader)):
        #if i > 2:
            #break 
        inputs = rearrange(inputs, 'b t h w c -> b c t h w').to(memory_format=torch.contiguous_format).float() 
        inputs = inputs.to(device)
        labels = labels.to(device)

        encoder_posterior = model.encode_first_stage(inputs)
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        xrec = model.decode_first_stage(z)
        xrec = torch.clamp((xrec+1.0)/2.0, min=0.0, max=1.0)

        num_frame = 8
        #import ipdb;ipdb.set_trace()
        for k in range(xrec.shape[0]):   #batch size
            video = (255. * rearrange(xrec[k].cpu().numpy(), 'b c h w -> b h w c')).astype(np.uint8)
        
            inputs_vid = torch.zeros(video.shape[0], 3, im_input_size, im_input_size)
            for ll in range(video.shape[0]):
                inputs_vid[ll] = transform(Image.fromarray(video[ll]))
            inputs_vid = inputs_vid.to(device)

            with torch.no_grad():
                outputs = model_ft(inputs_vid)
                preds = torch.round(nn.functional.sigmoid(outputs))
            all_predictions.append(preds.detach().cpu().numpy())
            labels_ = labels[k].repeat(num_frame,1)
            all_labels.append(labels_.cpu().numpy())
        
            ##statistics
            iter_corrects = torch.sum(preds == labels_.float().data)
            xidxs, yidxs = torch.where(labels_.data == 1)
            iter_recalls = sum([x.item() for x in [labels_.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
            total_positives += xidxs.size(0)
            for label, pred in zip(labels_, preds):
                if torch.all(torch.eq(label.float().data, pred)):
                    story_accuracy += 1
                for l, p in zip(label, pred):
                    if torch.all(torch.eq(l.float().data, p)):
                        image_accuracy += 1
            running_corrects += iter_corrects
            running_recalls += iter_recalls

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    preds = np.round(1 / (1 + np.exp(-all_predictions)))
    char_class = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
    print(classification_report(all_labels, all_predictions, target_names=char_class))
    print("Accuracy: ", accuracy_score(all_labels, preds))

    epoch_acc = float(running_corrects) * 100 / (all_labels.shape[0] * all_labels.shape[1])
    epoch_recall = float(running_recalls) * 100 / total_positives
    #print('Manually calculated accuracy: ', epoch_acc)
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, preds), epoch_recall))


def evaluate(image_path, ldm_model, model_name, model_path, mode):
    # Number of classes in the dataset
    num_classes = 6
    #   when True we only update the reshaped layer params
    feature_extract = False
    video_len = 5
    n_channels = 3

    phase = 'eval'

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model_ft.load_state_dict(torch.load(model_path))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    model_ft = model_ft.to(device)
    model_ft.eval()  # Set model to evaluate mode

    if image_path.endswith('.npy'):
        numpy_to_img(image_path, image_path[:-4], input_size)
        out_img_folder = image_path[:-4]
    else:
        out_img_folder = image_path
    input_size = 256
    # Create training and validation datasets
    image_dataset = StoryImageDataset(image_path, input_size, out_img_folder=out_img_folder, mode=mode)
    print("Number of samples in evaluation set: %s" % len(image_dataset))
    batch_size = 32

    # Create validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Number of batches in evaluation dataloader: %s" % len(dataloader))

    all_predictions = []
    all_labels = []
    story_accuracy = 0
    image_accuracy = 0

    running_corrects = 0
    running_recalls = 0
    total_positives = 0
    
    ######## Load pretrain model #########
    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    model = load_model_from_config(config, ldm_model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    ################ end #################
    scale = 5.0
    n_samples = 1
    n_iter = 1
    ddim_steps = 50
    ddim_eta = 0.0
    H = 256
    W = 256
    im_input_size = 299
    transform = transforms.Compose([
                transforms.Resize(im_input_size),
                transforms.CenterCrop(im_input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    # Iterate over data.
    for i, (_, inputs, _, labels, text_decs) in tqdm(enumerate(dataloader)):
        #inputs = inputs.view(batch_size * video_len, n_channels, inputs.shape[-2], inputs.shape[-1])
        #labels = labels.view(batch_size * video_len, labels.shape[-1])
        #inputs = inputs.to(device)
        labels = labels.to(device)
        num_vid = 4
        num_frame = 1
        #import ipdb; ipdb.set_trace()
        ######### video generation #########
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if scale != 1.0:
                    uc = torch.zeros(1, num_vid, 77, 1280).to(device)
                    for j in range(num_vid):
                        uc[:,j] = model.get_learned_conditioning(n_samples * [""])

                for k in range(len(text_decs)):
                    all_samples=list()
                    for n in trange(n_iter, desc="Sampling"):
                        c = torch.zeros(1, num_vid, 77, 1280).to(device)
                        batch_clip_video = text_decs[k].split(';')

                        for j in range(len(batch_clip_video)):
                            c[:,j] = model.get_learned_conditioning(n_samples * [batch_clip_video[j]])
 
                        shape = [4, num_vid, H//8, W//8]
                        samples_ddim, _ = sampler.sample(S=ddim_steps, conditioning=c, batch_size=n_samples, shape=shape, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta) 
                        x_samples_ddim = model.decode_first_stage(samples_ddim) 
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                         
                        for x_sample in x_samples_ddim:
                            x_sample_frame = (255. * rearrange(x_sample.cpu().numpy(), 'b c h w -> b h w c')).astype(np.uint8)
                            #import ipdb;ipdb.set_trace()
                            inputs = torch.zeros(x_sample_frame.shape[0], 3, im_input_size, im_input_size)
                            for ll in range(x_sample_frame.shape[0]):
                                inputs[ll] = transform(Image.fromarray(x_sample_frame[ll]))
                            inputs = inputs.to(device)
                            outputs = model_ft(inputs)  ### classification model
                            preds = torch.round(nn.functional.sigmoid(outputs))
                            all_predictions.append(preds.cpu().numpy())
                            all_labels.append(labels[k].repeat(num_vid,1).cpu().numpy())

                            ##statistics
                            labels_ = labels[k].repeat(num_frame,1)
                            iter_corrects = torch.sum(preds == labels_.float().data)
                            xidxs, yidxs = torch.where(labels_.data == 1)
                            iter_recalls = sum([x.item() for x in [labels_.float().data[xidx, yidx] == preds[xidx, yidx] for xidx, yidx in zip(xidxs, yidxs)]])
                            total_positives += xidxs.size(0)
                            for label, pred in zip(labels_, preds):
                                if torch.all(torch.eq(label.float().data, pred)):
                                    story_accuracy += 1
                                for l, p in zip(label, pred):
                                    if torch.all(torch.eq(l.float().data, p)):
                                        image_accuracy += 1
                            running_corrects += iter_corrects
                            running_recalls += iter_recalls

    #import ipdb;ipdb.set_trace()      
    ########      end      ##########

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(all_predictions.shape, all_labels.shape, image_accuracy, len(image_dataset))
    preds = np.round(1 / (1 + np.exp(-all_predictions)))
    char_class = ['Snow', 'Planet', 'Grass', 'Dirt', 'Stone', 'Sand']
    print(classification_report(all_labels, all_predictions, target_names=char_class))
    print("Accuracy: ", accuracy_score(all_labels, preds))

    epoch_acc = float(running_corrects) * 100 / (all_labels.shape[0] * all_labels.shape[1])
    epoch_recall = float(running_recalls) * 100 / total_positives
    #print('Manually calculated accuracy: ', epoch_acc)
    print('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, preds), epoch_recall))
    print('f1-score: {:.4f}'.format(f1_score(all_labels, preds, average='weighted', labels=np.unique(all_labels))))

    with open('result.txt', 'a') as f:
        f.write("eval_background_classifier\n")
        f.write('{} Acc: {:.4f} Recall: {:.4f}%'.format(phase, accuracy_score(all_labels, preds), epoch_recall))
        f.write('\nf1-score: {:.4f}'.format(f1_score(all_labels, preds, average='weighted', labels=np.unique(all_labels))))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Evaluate for Character Recall & InceptionScore')
    parser.add_argument('--image_path',  type=str, default='/ubc/cs/research/shield/datasets/coinrun/coinrun_dataset_jsons/release/')
    parser.add_argument('--ldm_model', type=str, default='logs/2023-01-30T13-49-14_txt2img-1p4B-train/checkpoints/epoch=000001.ckpt')
    parser.add_argument('--model_path', type=str, default='models/inception_32_0.0001/epoch-4.pt')
    parser.add_argument('--model_name', type=str, default='inception')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--ground_truth', action='store_true')
    args = parser.parse_args()

    if args.ground_truth:
        evaluate_gt(args.data_dir, args.model_name, args.model_path, args.mode)
    else:
        assert args.image_path, "Please enter the path to generated images"
        evaluate(args.image_path, args.ldm_model, args.model_name, args.model_path, args.mode)
        #evaluate_autoencoder(args.image_path, args.model_name, args.model_path, args.mode)
        #evaluate_gt(args.image_path, args.model_name, args.model_path, args.mode)

