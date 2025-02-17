%input_path /output_path 
load('F:\shujuji\3channel\shu3channel\mat\236.mat');
save_path='F:\code\gs\result\zzy\trainshu\signalchannel\1';
mkdir(fullfile(save_path,'迭代结果'));
%match prior
% load('F:\shujuji\3channel\shu3channel\mat\236.mat', 'original_image');
[rows, cols] = size(original_image);
imwrite(original_image,fullfile(save_path,'original_image.png'));
% get measurement;
proj = forward2(original_image);
save(fullfile(save_path,'proj.mat'),'proj');

% get sparse sampling image
sparse_image=backward2(proj);
sparse_image_max=max(sparse_image(:));
sparse_image_min=min(sparse_image(:));
sparse_image_normal=double((sparse_image-sparse_image_min)/(sparse_image_max-sparse_image_min));
imwrite(sparse_image_normal,fullfile(save_path,'DAS重建.png'));
save(fullfile(save_path,'DAS重建.mat'),'sparse_image_normal');
fid0 = fopen(fullfile(save_path,'measure.txt'),'a');
fprintf(fid0,'%.4f\r\n',psnr(original_image,sparse_image_normal));
fprintf(fid0,'%.4f\r\n',ssim(original_image,sparse_image_normal));
fclose(fid0);

% params of pwls
beta = 0.005;
pwls_iter = 4;
iter = 700;
pwls = zeros(size(original_image));
reconstruction = zeros(size(original_image));
maxvalue_exchange = 255/max(original_image(:));

% load net set params 

load('F:\code\gs\modelsa\shu\natural_image_3channel_3_v1\nature3natural_image_3channel_3_v1-epoch-38.mat');
net1 = dagnn.DagNN.loadobj(net) ;
net1.removeLayer('objective') ;
out_idx1 = net1.getVarIndex('prediction') ;
net1.vars(net1.getVarIndex('prediction')).precious = 1 ;
net1.mode = 'test';
net1.move('gpu');
sigma_net1 = 3;


clear net;
load('F:\code\gs\modelsa\shu\natural_image_3channel_8_v1\nature8natural_image_3channel_8_v1-epoch-39.mat');
net2 = dagnn.DagNN.loadobj(net) ;
net2.removeLayer('objective') ;
out_idx2 = net2.getVarIndex('prediction') ;
net2.vars(net2.getVarIndex('prediction')).precious = 1 ;
net2.mode = 'test';
net2.move('gpu');
sigma_net2 = 8;

clear net;
load('F:\code\gs\modelsa\shu\natural_image_3channel_15_v1\nature15natural_image_3channel_15_v1-epoch-38.mat');
net3 = dagnn.DagNN.loadobj(net) ;
net3.removeLayer('objective') ;
out_idx3 = net3.getVarIndex('prediction') ;
net3.vars(net3.getVarIndex('prediction')).precious = 1 ;
net3.mode = 'test';
net3.move('gpu');
sigma_net3 = 15;

% 定义变量输出文件夹
% output_folders = {'input1',  'rec', 'input2','rec1',  'input3', 'rec2','re0',  're1', 're2'};
              
% for k = 1:length(output_folders)
%     folder_path = fullfile(save_path, output_folders{k});
%     if ~exist(folder_path, 'dir')
%         mkdir(folder_path);
%     end
% end
% clear net;
% load('F:\code\gs\modelsa\qiu\natural_image_3channel_25_v1\nature25natural_image_3channel_25_v1-epoch-39.mat');
% net4 = dagnn.DagNN.loadobj(net) ;
% net4.removeLayer('objective') ;
% out_idx4 = net4.getVarIndex('prediction') ;
% net4.vars(net4.getVarIndex('prediction')).precious = 1 ;
% net4.mode = 'test';
% net4.move('gpu');
% sigma_net4 = 25;
% 
% clear net;
% load('F:\code\gs\modelsa\qiu\natural_image_3channel_50_v1\nature50natural_image_3channel_50_v1-epoch-35.mat');
% net5 = dagnn.DagNN.loadobj(net) ;
% net5.removeLayer('objective') ;
% out_idx5 = net5.getVarIndex('prediction') ;
% net5.vars(net5.getVarIndex('prediction')).precious = 1 ;
% net5.mode = 'test';
% net5.move('gpu');
% sigma_net5 = 50;

%迭代过程
for i = 1 : iter
    % pwls reconstruction
    if i < 40
    pwls = split_hscg1(reconstruction, proj, reconstruction, beta, pwls_iter);
    else
        pwls = split_hscg1(pwls, proj, reconstruction,beta, pwls_iter);
    end
    pwls(pwls < 0) = 0;
    
    %% descent gradient of REDAEP
    
    input = repmat(pwls*maxvalue_exchange ,[1,1,3]);
    
    noise = randn(size(input)) * sigma_net1;
    input1=single((input+noise)/255);
    rec=Processing_Im_w1(input1, net1, 1, out_idx1);
    rec = double(rec);
    prior_err0 = input - rec;
%     re0 = rec/ maxvalue_exchange;
    prior_err0_1 = prior_err0 / maxvalue_exchange;

        % 保存网络1的中间结果
%     imwrite(input1, fullfile(save_path, 'input1', sprintf('%d.png', i)));
%     imwrite(rec, fullfile(save_path, 'rec', sprintf('%d.png', i)));
%     imwrite(re0, fullfile(save_path, 're0', sprintf('%d.png', i)));

    noise = randn(size(input)) * sigma_net2;
    input2=single((input+noise)/255);
    rec1=Processing_Im_w1(input2, net2, 1, out_idx2);
    rec1 = double(rec1);
    prior_err1 = input - rec1;
%     re1 = rec1/ maxvalue_exchange;
    prior_err1_1 = prior_err1 / maxvalue_exchange;

        % 保存网络2的中间结果
%     imwrite(input2, fullfile(save_path, 'input2', sprintf('%d.png', i)));
%     imwrite(rec1, fullfile(save_path, 'rec1', sprintf('%d.png', i)));
%     imwrite(re1, fullfile(save_path, 're1', sprintf('%d.png', i)));
      
    noise = randn(size(input)) * sigma_net3;
    input3=single((input+noise)/255);
    rec2=Processing_Im_w1(input3, net3, 1, out_idx3);
    rec2 = double(rec2);
    prior_err2 = input - rec2;
%     re2 = rec2/ maxvalue_exchange;
    prior_err2_1 = prior_err2 / maxvalue_exchange;
    
        % 保存网络3的中间结果
%     imwrite(input3, fullfile(save_path, 'input3', sprintf('%d.png', i)));
%     imwrite(rec2, fullfile(save_path, 'rec2', sprintf('%d.png', i)));
%     imwrite(re2, fullfile(save_path, 're2', sprintf('%d.png', i)));
% %     
%     noise = randn(size(input)) * sigma_net4;
%     input4=single((input+noise)/255);
%     rec3=Processing_Im_w1(input4, net4, 1, out_idx4);
%     rec3 = double(rec3);
%     prior_err3 = input - rec3;
%     prior_err3_1 = prior_err3 / maxvalue_exchange;
%     
%     noise = randn(size(input)) * sigma_net5;
%     input5=single((input+noise)/255);
%     rec4=Processing_Im_w1(input5, net5, 1, out_idx5);
%     rec4 = double(rec4);
%     prior_err4 = input - rec4;
%     prior_err4_1 = prior_err4 / maxvalue_exchange;
     
%     五模型    
%     prior_err_final=(prior_err0_1+prior_err1_1+prior_err2_1+prior_err3_1+prior_err4_1)/5;
%     prior_err_final=mean(prior_err_final,3);
%     reconstruction = double(pwls - prior_err_final);
%     三模型    
    prior_err_final=(prior_err0_1+prior_err1_1+prior_err2_1)/3;
%     prior_err_final=mean(prior_err_final,3);
    prior_err_final1 = prior_err_final(:, :, 1);  % 对应 R 通道
    reconstruction = double(pwls - prior_err_final1);
    
%     双模型    
%     prior_err_final=(prior_err0_1+prior_err1_1)/2;
%     prior_err_final=mean(prior_err_final,3);
%     reconstruction = double(pwls - prior_err_final);
%     单模型   
%     prior_err_final=prior_err2_1;
%     prior_err_final=mean(prior_err_final,3);
%     reconstruction = double(pwls - prior_err_final); 

%     reconstruction=reconstruction(:,:,1);
    reconstruction(reconstruction < 0) = 0;
    
    %save reconstruction and evaluate result
    png_save=['迭代结果\',num2str(i),'.png'];   
    imwrite(reconstruction,fullfile(save_path,png_save));
    fid = fopen(fullfile(save_path,'psnr.txt'),'a');
    fprintf(fid,'%.4f\r\n',psnr(original_image,reconstruction));
    fclose(fid);
    fid1 = fopen(fullfile(save_path,'ssim.txt'),'a');
    fprintf(fid1,'%.4f\r\n',ssim(original_image,reconstruction));
    fclose(fid1);

end