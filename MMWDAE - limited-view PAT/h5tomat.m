clear
close all;

load filenames

jj=0;
for i = 1 : length(filenames) %%≤‚ ‘“ªœ¬1402
    
    filename = filenames{i};
    [substr,~]=strsplit(filename,'.');
    fig_index_str = strsplit(substr{1},'file100');
    fig_index = str2num(fig_index_str{1,2});
    
    knee_reconstructions = h5read(filename,'/reconstruction');
    for ii = [15,18]
        knee_reconstruction = knee_reconstructions(:,:,ii);
        jj = jj+1;
        save (['reconstructions_val/knee_reconstruction_',num2str(jj,'%02d')],'knee_reconstruction');
        figure(10000*(ii-12)/(18-12)+fig_index);imshow(abs(knee_reconstruction),[]);
    end    
end


