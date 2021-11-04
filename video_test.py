from model.blindsr import BlindSR
import torch
import numpy as np
import imageio
import argparse
import os
import utility
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',
                        type=str,
                        default='./dataset/v_ApplyEyeMakeup_g05_c01/image_1.jpg',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='2', help='super resolution scale')
    parser.add_argument('--resume', type=int, default=600, help='resume from specific checkpoint')
    parser.add_argument('--blur_type',
                        type=str,
                        default='iso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()

def make_video(H, W, image_name, image_num, scale):
    size = (H,W)#这个是图片的尺寸，一定要和要用的图片size一致
    #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowrite = cv2.VideoWriter('./video_results/' + image_name + '_sr_x' + str(scale) + '.avi', fourcc, 25, size)#20是帧数，size是图片尺寸
    img_array=[]
    for filename in ['./results/' + image_name + '/x' + str(scale) + '/image_{0}_sr.png'.format(i+1) for i in range(image_num)]:#这个循环是为了读取所有要用的图片文件
        img = cv2.imread(filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)
    for i in range(len(img_array)):#把读取的图片文件写进去
        videowrite.write(img_array[i])
    videowrite.release()
    print('end!')

def sr_image(image_name, image_num, image_path):
    args = parse_args()
    print(args)
    # path to save sr images
    save_dir = './results/' + image_name + '/x' + str(args.scale[0])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    dir = './experiment/blindsr_x' + str(args.scale[0]) + '_bicubic_iso'
    DASR = BlindSR(args).cuda()
    DASR.load_state_dict(torch.load(dir + '/model/model_' + str(args.resume) + '.pt'), strict=False)
    DASR.eval()
    # print(args.img_dir)
    
    for i in range(image_num):
        m = i+1
        img_dir = image_path + '/image_' + str(m) + '.png'
        # print(img_dir)
        lr = imageio.imread(img_dir)
        lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
        lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

        # inference
        sr = DASR(lr[:, 0, ...])
        sr = utility.quantize(sr, 255.0)

        # save sr results
        img_name = img_dir.split('.png')[0].split('/')[-1]
        sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
        sr = sr[:, :, [2, 1, 0]]
        print(img_name)
        cv2.imwrite(save_dir + '/' + img_name  + '_sr.png', sr)

    # make_video(640, 480, image_name, image_num, args.scale[0])
    H = 320 * int(args.scale[0])
    W = 240 * int(args.scale[0])
    make_video(H, W, image_name, image_num, int(args.scale[0]))

def main():
    
    image_name = 'v_BandMarching_g05_c06'
    image_num = 79
    image_path = './dataset/' + image_name

    sr_image(image_name, image_num, image_path)

    # for scale in range(2, 4):
    #     print("x %d: begin", scale)
    #     sr_image(image_name, image_num, image_path, scale, args)
    #     print("x %d: end", scale)


if __name__ == '__main__':
    with torch.no_grad():
        main()
