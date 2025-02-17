% for n=1:20
%     a=['E:\光声断层成像\复现\REDAEP\REDAEP-master\REDAEP-master\Test_Images\批量训练\小鼠\',num2str(n),'.mat'];
%     load(a);
%     b=['E:\光声断层成像\复现\REDAEP\REDAEP-master\REDAEP-master\Test_Images\批量训练\小鼠\图片\',num2str(n),'.png'];
%     imwrite(original_image,b);
% end
for n=100:800
 a=['E:\光声断层成像\复现\MWCNN2020-master最原始\Training data\DIV2K_train_HR\DIV2K_train_HR\0',num2str(n),'.png'];
RGB = imread(strcat(a));  % 读入需要转换的彩色图片
I = rgb2gray(RGB);   % 转换语法
gray=imresize(I,[256,256]);
b=['E:\光声断层成像\复现\MWCNN2020-master最原始\Training data\Training_gray\',num2str(n-9),'.png'];
imwrite(gray,b)  % 保存转换成功的灰度图
end