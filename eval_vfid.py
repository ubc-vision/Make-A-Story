import torchvision.transforms as transforms
import argparse
import os, sys
from classifier.eval_dataloader_multi import StoryImageDataset
sys.path.append('.')

from classifier.fid_score import fid_score

def main(args):

    image_transforms = transforms.Compose([
        transforms.Resize((args.imsize, args.imsize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    ref_dataset = StoryImageDataset(args.img_ref_dir,
                                    args.imsize,
                                    mode=args.mode,
                                    out_img_folder=None, fid = True)

    fid = fid_score(ref_dataset, args.model_path, cuda=True, normalize=True, batch_size=1)
    print('Frechet Image Distance: ', fid)

    with open('result.txt', 'a') as f:
        f.write("\nFID-score: "+ str(fid))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Frechet Story and Image distance')
    parser.add_argument('--img_ref_dir', type=str, default='/ubc/cs/research/shield/datasets/coinrun/coinrun_dataset_jsons/release/')
    parser.add_argument('--model_path', type=str, default='logs/2023-01-30T13-49-14_txt2img-1p4B-train/checkpoints/epoch=000001.ckpt')
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--imsize', type=int, default=256)
    args = parser.parse_args()

    print(args)
    main(args)
