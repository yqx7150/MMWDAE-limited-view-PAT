% ����ͼƬ
image_data = imread('F:\code\gs\result\zzy\real\UBP\meat\1.png'); % �滻 your_image.jpg Ϊ��Ҫ�����ͼƬ�ļ���
image_data = im2double(image_data);
image_data_max=max(image_data(:));
image_data_min=min(image_data(:));
original_image=double((image_data-image_data_min)/(image_data_max-image_data_min));

% original_image = image_data;
% original_image = fliplr(original_image); %���ҷ�ת
% original_image = imrotate(original_image, -90);
% imwrite(original_image,'F:\shujuji\3channel\xue3channnel\test\5.png'); 

% ����һ������������ͼ��ͼ�δ���
figure;
% ��ʾԭʼͼ��
imshow(original_image);

% ����ͼ�����ݵ�.mat�ļ���
save(fullfile('F:\code\gs\result\zzy\real\UBP\meat','1.mat'),'original_image');

% imwrite(original_image,fullfile('F:\code\gs\Test_Images\train\xue','23.png'));
