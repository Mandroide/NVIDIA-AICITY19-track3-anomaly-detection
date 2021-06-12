# -*- coding: utf-8 -*-
import argparse
import itertools
import pathlib

import cv2

from SOT.utils import natural_keys


parser = argparse.ArgumentParser(description='This program produces background and foreground frames for every video.')
parser.add_argument('video_path', help='Directory with the MP4 or AVI videos.', type=pathlib.Path)
video_path: pathlib.Path = parser.parse_args().video_path

if video_path.is_dir():
    save_bg_path = video_path.with_name(video_path.stem + '_bg_imgs')
    save_fg_path = video_path.with_name(video_path.stem + '_fg_imgs')

    #path = os.path.join(t, 'videos')
    #videos: typing.Generator[pathlib.Path] = video_path.glob("*.mp4")
    videos = sorted(itertools.chain(video_path.glob('*.avi'), video_path.glob('*.mp4')), key=natural_keys)

    for v in videos:
        v_name = str(v)
        print("Now for {}".format(v.name))
        save_bg_path_ = save_bg_path/v.stem
        save_fg_path_ = save_fg_path/v.stem
        save_fg_path_.mkdir(parents=True, exist_ok=True)
        save_bg_path_.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(v_name)

        bg = cv2.createBackgroundSubtractorMOG2()
        bg.setHistory(120)

        #fourcc = cv2.VideoWriter_fourcc(*'FLV1')
        ret, frame = cap.read()
        #h, w, _ = frame.shape
        #fg_writer = cv2.VideoWriter(
        #    os.path.join(save_fg_path,'{}.flv'.format(v.split('.')[0])),
        #    fourcc, 30, (w,h))
        count = 0

        while ret:
            fg_img = bg.apply(frame)
            #fg_img_rgb = np.expand_dims(fg_img, axis=2)
            #fg_img_rgb = np.concatenate((fg_img, fg_img, fg_img), axis=-1)
            #fg_writer.write(fg_img_rgb)
            filename = save_fg_path_/'{}.png'.format(count)
            cv2.imwrite(str(filename), fg_img)
            bg_img = bg.getBackgroundImage()
            #if count%30==0:
            filename = save_bg_path_ / '{}.png'.format(count)
            cv2.imwrite(filename, bg_img)
            ret, frame = cap.read()
            count += 1
        cap.release()
        #fg_writer.release()
else:
    raise ValueError("The video_path is not a directory.")
