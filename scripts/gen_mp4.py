import os
import cv2

# 文件夹路径
folder_path = '/home/nfs/jsh/DisCo/eval/eval_pt3.0_1_dino_hd_try2.0_fix_fat_vedio3/pred_image'  # 替换为实际的文件夹路径

# 读取文件夹中的图片
image_files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png')])
# print(image_files)
# 创建一个图像列表
images = []
for i in range(len(image_files)):
    # if 2<=i and i<9:
    image_file = image_files[i]
    images.append(cv2.imread(image_file))

height, width, _ = images[0].shape
mp4_path = '/home/nfs/jsh/DisCo/eval/eval_pt3.0_1_dino_hd_try2.0_fix_fat_vedio3/pred_image/output.mp4'  # 替换为实际的输出路径
frame_rate = 2.0
# 创建视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(mp4_path, fourcc, frame_rate, (width, height))

# 将图像逐帧写入视频
for image in images:
    video_writer.write(image)

# 释放视频编写器
video_writer.release()

print(f"MP4 视频已保存至 {mp4_path}")
