Femur = [255 0 0] ;   
Tibia = [128 0 128] ;    
Cartilage_up = [0 255 0] ;   
Cartilage_dowm = [255 255 0];   
BackGround = [0 0 0];
save_dirs_path='E:\zys\GAN\datasets\OAI\test';
if ~exist(save_dirs_path,'dir')
    mkdir(save_dirs_path);
end
files_path='E:\zys\OAI\test_img';

files_mask_path='E:\zys\OAI\test_mask';

files=dir(files_path);

files_len=length(files);
for i=3:files_len
    
    imgs_path=strcat(files_path,'\',files(i).name);
    imgs_mask_path=strcat(files_mask_path,'\',files(i).name);
   
    imgs=dir(imgs_path);
    sort_img_name=sort_nat({imgs.name});
   
    imgs_len=length(imgs);
    for j=3:imgs_len
       
        img_path=strcat(imgs_path,'\',sort_img_name{j});
        img=imread(img_path);
        %img=double(img);
       % img=img/255;
        
        img = single(img) / 255;  
        
        img_mask_path=strcat(imgs_mask_path,'\',sort_img_name{j});
        img_mask=imread(img_mask_path);
        [W,H,C] = size(img_mask);
        mat_6 = zeros(384,384,6,'single');
        num=0;
        for w = 1:W
            for h = 1:H
                o = reshape(img_mask(w,h,:),1,3);
                mat_6(w,h,6) = img(w,h);
                if isequal(o,Femur)     
                    mat_6(w,h,1) = 1;
                elseif isequal(o,Tibia)   
                    mat_6(w,h,2) = 1;
                elseif isequal(o,Cartilage_up)  
                    mat_6(w,h,3) = 1;
                elseif isequal(o,Cartilage_dowm)  
                    mat_6(w,h,4) = 1;         
                %elseif isequal(o,Muscle)
                %    mat_5(w,h,3) = 1;
                %elseif isequal(o,Fat)
                 %   mat_5(w,h,5) = 1;
                %elseif isequal(o,Ligament)
                %    mat_5(w,h,6) = 1;
                %elseif isequal(o,MeniscusInjury)
                    %mat_5(w,h,7) = 1;
                elseif isequal(o,BackGround)
                    mat_6(w,h,5) = 1;
                else
                
                    num = num+1;
                    fprintf('%d_%d_%d\n',o(1),o(2),o(3));
                end
            end
        end 
        if num ~= 0
            fprintf('%s.%s',files(i).name,sort_img_name{j});
        end
        save_dir_path=strcat(save_dirs_path,'\',files(i).name);
        if ~exist(save_dir_path,'dir')
            mkdir(save_dir_path);
        end     
        save_path = strcat(save_dir_path,'\',int2str(j-2),'.mat');
%         save_path = sprintf('D:\\renhongjin\\GAN_att0_10\\datasets\\DATA10_seg200_3xueguan\\%s\\%d.mat',files(i).name,j-2);
        %save_path=strcat(save_dir_path,'\',(j-2),'.mat')
        save(save_path,'mat_6');
        clear mat_6;
    end
end