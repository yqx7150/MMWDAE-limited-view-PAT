% for n=1:20
%     a=['E:\�����ϲ����\����\REDAEP\REDAEP-master\REDAEP-master\Test_Images\����ѵ��\С��\',num2str(n),'.mat'];
%     load(a);
%     b=['E:\�����ϲ����\����\REDAEP\REDAEP-master\REDAEP-master\Test_Images\����ѵ��\С��\ͼƬ\',num2str(n),'.png'];
%     imwrite(original_image,b);
% end
for n=100:800
 a=['E:\�����ϲ����\����\MWCNN2020-master��ԭʼ\Training data\DIV2K_train_HR\DIV2K_train_HR\0',num2str(n),'.png'];
RGB = imread(strcat(a));  % ������Ҫת���Ĳ�ɫͼƬ
I = rgb2gray(RGB);   % ת���﷨
gray=imresize(I,[256,256]);
b=['E:\�����ϲ����\����\MWCNN2020-master��ԭʼ\Training data\Training_gray\',num2str(n-9),'.png'];
imwrite(gray,b)  % ����ת���ɹ��ĻҶ�ͼ
end