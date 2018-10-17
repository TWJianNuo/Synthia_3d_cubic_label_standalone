% Helper, deal with acquiring corresponding data from different data set
function path_info = helper()
    output_father_path = make_dir_out_re_father(); inter_father_path = make_dir_int_re_father();
    synthia_names = {'SYNTHIA-SEQS-05-SPRING';};
    path_info = cell(length(synthia_names),1);
    for i = 1 : length(synthia_names)
        path_info{i} = generate_path_info(synthia_names{i}, output_father_path, inter_father_path);
    end
    %{
    path_info{1,1} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/'; % base file path
    path_info{1,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{1,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{1,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{1,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{1,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{1,7} = 294;
    path_info{1,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';

    path_info{2,1} = '/home/ray/Downloads/SYNTHIA-SEQS-04-SPRING/'; % base file path
    path_info{2,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{2,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{2,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{2,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{2,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{2,7} = 958;
    path_info{2,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';

    path_info{3,1} = '/home/ray/Downloads/SYNTHIA-SEQS-05-SUNSET/'; % base file path
    path_info{3,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{3,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{3,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{3,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{3,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{3,7} = 707;
    path_info{3,8} = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYTHIA_Others/';
    %}
end
function path_info = generate_path_info(synthia_name, output_father_path, inter_father_path)
    % Code file, Synthia source data file, support data file and output data file
    current_folder_path = pwd;
    str_com = strsplit(current_folder_path, '/'); 
    father_folder = zeros(0);
    for i = 1 : length(str_com) - 1
        father_folder = [father_folder str_com{i} '/'];
    end

    path_info{1,1} = [father_folder 'Synthia_data/' '/' synthia_name '/']; % base file path
    path_info{1,2} = 'Depth/Stereo_Left/Omni_F/'; % depth file path
    path_info{1,3} = 'GT/LABELS/Stereo_Left/Omni_F/'; % Segmentation mark path
    path_info{1,4} = 'RGB/Stereo_Left/Omni_F/';
    path_info{1,5} = 'GT/COLOR/Stereo_Left/Omni_F/';
    path_info{1,6} = 'CameraParams/Stereo_Left/Omni_F/';
    path_info{1,7} = get_num_img_in_folder(path_info);
    path_info{1,8} = make_dir_out_re(output_father_path, synthia_name);
    path_info{1,9} = make_dir_inter_re(inter_father_path, synthia_name);
end
function inter_path = make_dir_inter_re(inter_father_path, synthia_name)
    inter_path = [inter_father_path synthia_name '/'];
    mkdir(inter_path);
end
function output_path = make_dir_out_re(output_father_path, synthia_name)
    output_path = [output_father_path synthia_name '/'];
    mkdir(output_path);
end
function inter_father_path = make_dir_int_re_father()
    current_folder_path = pwd;
    str_com = strsplit(current_folder_path, '/');
    father_folder = zeros(0);
    for i = 1 : length(str_com) - 1
        father_folder = [father_folder str_com{i} '/'];
    end
    mkdir([father_folder 'inter_results/']); inter_father_path = [father_folder 'inter_results/'];
end
function output_father_path = make_dir_out_re_father()
    current_folder_path = pwd;
    str_com = strsplit(current_folder_path, '/');
    father_folder = zeros(0);
    for i = 1 : length(str_com) - 1
        father_folder = [father_folder str_com{i} '/'];
    end
    mkdir([father_folder 'output_results/']); output_father_path = [father_folder 'output_results/'];
end
function num = get_num_img_in_folder(path_info)
    name_cluster = dir([path_info{1} path_info{4} '*.png']);
    num = size(name_cluster,1) - 1;
end