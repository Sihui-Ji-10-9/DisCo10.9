import os
import imageio

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
    images.append(imageio.imread(image_file))

# 设置 GIF 参数
gif_path = '/home/nfs/jsh/DisCo/eval/eval_pt3.0_1_dino_hd_try2.0_fix_fat_vedio3/pred_image/output.gif'  # 替换为实际的输出路径
gif_duration = 0.5  # 每张图片的显示时间（秒）

# 保存为 GIF 视频
imageio.mimsave(gif_path, images, duration=gif_duration)

print(f"GIF 视频已保存至 {gif_path}")