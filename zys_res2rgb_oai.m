clear all;clc;

Femur = [255 0 0] ;   
Tibia = [128 0 128] ;    
Cartilage_up = [0 255 0] ;    
Cartilage_dowm = [255 255 0];  
BackGround = [0 0 0];


path1 = 'E:\zys\GAN\GAN_results\gan_seg_384\test_result';
save_path = 'E:\zys\result\OAI\rgb';

if ~exist(save_path,'dir')
    mkdir(save_path)
end
dirs=dir(path1);
dirs_len=length(dirs);
for i=3:dirs_len
    dir_path=strcat(path1,'\',dirs(i).name);
    dir_save_path=strcat(save_path,'\',dirs(i).name);
    if ~exist(dir_save_path,'dir')
        mkdir(dir_save_path);
    end
    [a b] = strread(dirs(i).name, '%s %s','delimiter', '_');
    images=dir(dir_path);
    sort_images_name=sort_nat({images.name}); 
    image_len=length(images);

    for j=3:image_len
        image_path=strcat(dir_path,'\',sort_images_name{j});
        image=imread(image_path);
        res=uint8(zeros(384,384,3));
        for m=1:384
            for n=385:768
                if image(m,n)~=0
                    res(m,n-384,:)= Femur;
                end
            end
        end
        for m=385:768
            for n=385:768
                if image(m,n)~=0
                    res(m-384,n-384,:)= Tibia;
                end
            end
        end
        for m=769:1152
            for n=385:768
                if image(m,n)~=0
                    res(m-768,n-384,:)= Cartilage_up;
                end
            end
        end
        for m=1153:1536
            for n=385:768
                if image(m,n)~=0
                    res(m-1152,n-384,:)= Cartilage_dowm;
                end
            end
        end
        for m=1537:1920
            for n=385:768
                if image(m,n)~=0
                    res(m-1536,n-384,:)= BackGround;
                end
            end
        end
     
    %    for m=3073:3456
     %       for n=385:768
      %          if image(m,n)~=0
       %             res(m-3072,n-384,:)= BackGround;
        %        end
         %   end
       % end            
        res_save_path=strcat(dir_save_path,'\',sort_images_name{j});
        imwrite(res,res_save_path);
    end
    
end
