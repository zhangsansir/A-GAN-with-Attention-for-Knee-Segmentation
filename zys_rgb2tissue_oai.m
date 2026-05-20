clear all;clc;

Femur = [255 0 0] ;    
Tibia = [128 0 128] ;    
Cartilage_up = [0 255 0] ;   
Cartilage_dowm = [255 255 0];   
BackGround = [0 0 0];

raw_size=384;
col_size=384;
rgb_path='E:\zys\result_test\OAI\rgb';
tissuse_num=3;
save_path='E:\zys\result_test\OAI\tissue';
if ~exist(save_path,'dir')
    mkdir(save_path)
end

rgb_dirs=dir(rgb_path);
dirs_len=length(rgb_dirs);
for i=3:dirs_len
    str=split(rgb_dirs(i).name,'-');
    person_name=str{1};
    rgb_dir_path=strcat(rgb_path,'\',rgb_dirs(i).name);    
    dir_Femur_save_path=strcat(save_path,'\Femur\',person_name);
    if ~exist(dir_Femur_save_path,'dir')
        mkdir(dir_Femur_save_path);
    end
    dir_Tibia_save_path=strcat(save_path,'\Tibia\',person_name);
    if ~exist(dir_Tibia_save_path,'dir')
        mkdir(dir_Tibia_save_path);
    end
    dir_Cartilage_up_save_path=strcat(save_path,'\Cartilage_up\',person_name);
    if ~exist(dir_Cartilage_up_save_path,'dir')
        mkdir(dir_Cartilage_up_save_path);
    end    
    dir_Cartilage_dowm_save_path=strcat(save_path,'\Cartilage_dowm\',person_name);
    if ~exist(dir_Cartilage_dowm_save_path,'dir')
        mkdir(dir_Cartilage_dowm_save_path);
    end
    %dir_CorticalBone_save_path=strcat(save_path,'\CorticalBone\',person_name);
    %if ~exist(dir_CorticalBone_save_path,'dir')
     %   mkdir(dir_CorticalBone_save_path);
    %end    
    %dir_SpongyBone_save_path=strcat(save_path,'\SpongyBone\',person_name);
    %if ~exist(dir_SpongyBone_save_path,'dir')
    %    mkdir(dir_SpongyBone_save_path);
    %end   
    %dir_MeniscusInjury_save_path=strcat(save_path,'\MeniscusInjury\',person_name);
    %if ~exist(dir_MeniscusInjury_save_path,'dir')
     %   mkdir(dir_MeniscusInjury_save_path);
    %end
    %dir_Cartilage_save_path=strcat(save_path,'\Cartilage\',person_name);
    %if ~exist(dir_Cartilage_save_path,'dir')
     %   mkdir(dir_Cartilage_save_path);
   % end   
    rgb_imgs=dir(rgb_dir_path);
    sort_rgbimgs_name=sort_nat({rgb_imgs.name});
    imgs_len=length(rgb_imgs);
    for j = 3: imgs_len
        rgb_img_path=strcat(rgb_dir_path,'\',sort_rgbimgs_name{j});
        rgb_img = imread(rgb_img_path);
        Femur_clip=zeros(raw_size,col_size);
        Tibia_clip = zeros(raw_size,col_size);
        Cartilage_up_clip=zeros(raw_size,col_size);
        Cartilage_dowm_clip = zeros(raw_size,col_size);
       % SpongyBone_clip = zeros(raw_size,col_size);
       % CorticalBone_clip=zeros(raw_size,col_size);
       % Vascular_clip = zeros(raw_size,col_size);
       % Ligament_clip = zeros(raw_size,col_size);    
        for w = 1:raw_size
            for h = 1:col_size
                o = reshape(rgb_img(w,h,:),1,3);
                if isequal(o,Femur)
                     Femur_clip(w,h) = 255;
                elseif isequal(o,Tibia)
                    Tibia_clip(w,h) = 255;
                elseif isequal(o,Cartilage_up)
                    Cartilage_up_clip(w,h) = 255;
                elseif isequal(o,Cartilage_dowm)
                    Cartilage_dowm_clip(w,h) = 255;
                %elseif isequal(o,SpongyBone)
                 %   SpongyBone_clip(w,h) = 255;
                %elseif isequal(o,CorticalBone)
                 %   CorticalBone_clip(w,h) = 255;
             %   elseif isequal(o,Vascular)
             %       Vascular_clip(w,h) = 255;
             %   elseif isequal(o,Vascular_python)
             %       Vascular_clip(w,h) = 255;
               % elseif isequal(o,Ligament)
                %    Ligament_clip(w,h) = 255;    
               % elseif isequal(o,Ligament_python)
                %    Ligament_clip(w,h) = 255;
                end
            end
        end
        
%           for w = 93 : 292
%             for h = 1:col_size
%                 o = reshape(rgb_img(w,h,:),1,3);
%                 if isequal(o,Muscle)
%                     Muscle_clip(w-92,h) = 255;
%                 elseif isequal(o,Fat)
%                     Fat_clip(w-92,h) = 255;
%                 elseif isequal(o,Cartilage)
%                     Cartilage_clip(w-92,h) = 255;
%                 elseif isequal(o,MeniscusInjury)
%                     MeniscusInjury_clip(w-92,h) = 255;
%                 elseif isequal(o,SpongyBone)
%                     SpongyBone_clip(w-92,h) = 255;
%                 elseif isequal(o,CorticalBone)
%                     CorticalBone_clip(w-92,h) = 255;
%                 elseif isequal(o,Vascular)
%                     Vascular_clip(w-92,h) = 255;
%                 elseif isequal(o,Vascular_python)
%                     Vascular_clip(w-92,h) = 255;
%                 elseif isequal(o,Ligament)
%                     Ligament_clip(w-92,h) = 255;    
%                 elseif isequal(o,Ligament_python)
%                     Ligament_clip(w-92,h) = 255;
%                 end
%             end
%         end
        save_Femur_path=strcat(dir_Femur_save_path,'\',int2str(j-2),'.bmp');
        imwrite(Femur_clip,save_Femur_path);
        save_Tibia_path=strcat(dir_Tibia_save_path,'\',int2str(j-2),'.bmp');
        imwrite(Tibia_clip,save_Tibia_path);
        save_Cartilage_up_path=strcat(dir_Cartilage_up_save_path,'\',int2str(j-2),'.bmp');
        imwrite(Cartilage_up_clip,save_Cartilage_up_path);
        save_Cartilage_dowm_path=strcat(dir_Cartilage_dowm_save_path,'\',int2str(j-2),'.bmp');
        imwrite(Cartilage_dowm_clip,save_Cartilage_dowm_path);
      %  save_SpongyBone_path=strcat(dir_SpongyBone_save_path,'\',int2str(j-2),'.bmp');
       % imwrite(SpongyBone_clip,save_SpongyBone_path);
       % save_CorticalBone_path=strcat(dir_CorticalBone_save_path,'\',int2str(j-2),'.bmp');
        %imwrite(CorticalBone_clip,save_CorticalBone_path);
       % save_Vascular_path=strcat(dir_Vascular_save_path,'\',int2str(j-2),'.bmp');
       % imwrite(Vascular_clip,save_Vascular_path);
        %save_Ligament_path=strcat(dir_Ligament_save_path,'\',int2str(j-2),'.bmp');
        %imwrite(Ligament_clip,save_Ligament_path);
        
    end
end

        
                    
                
        
        
    