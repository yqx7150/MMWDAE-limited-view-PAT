% 加载图片
image_data = imread('F:\code\gs\result\zzy\real\UBP\meat\1.png'); % 替换 your_image.jpg 为你要处理的图片文件名
image_data = im2double(image_data);
image_data_max=max(image_data(:));
image_data_min=min(image_data(:));
original_image=double((image_data-image_data_min)/(image_data_max-image_data_min));

% original_image = image_data;
% original_image = fliplr(original_image); %左右翻转
% original_image = imrotate(original_image, -90);
% imwrite(original_image,'F:\shujuji\3channel\xue3channnel\test\5.png'); 

% 创建一个包含两个子图的图形窗口
figure;
% 显示原始图像
imshow(original_image);

% 保存图像数据到.mat文件中
save(fullfile('F:\code\gs\result\zzy\real\UBP\meat','1.mat'),'original_image');

% imwrite(original_image,fullfile('F:\code\gs\Test_Images\train\xue','23.png'));
