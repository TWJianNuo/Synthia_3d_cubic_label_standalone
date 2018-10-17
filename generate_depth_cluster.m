function generate_depth_cluster(help_info)
    for jj = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, max_frame_num, save_path, inter_path] = read_helper_info(help_info, jj);
        depth_cluster = init_depth_cluster();
        for i = 1 : max_frame_num
            data_cluster = read_in_data_cluster(i, help_info{jj});
            cur_depth_map = read_in_depth_map(i, base_path, GT_Depth_path);
            depth_cluster = merge_depth_cluster(cur_depth_map, depth_cluster, i);
            depth_cluster = clean_depth_cluster(data_cluster, depth_cluster);
            save_depth_cluster(i, depth_cluster, help_info{jj});
        end
    end
end
function earliest_frame = find_earliest_frame_num(data_cluster)
    earliest_frame = 100000;
    for i = 1 : length(data_cluster)
        if earliest_frame > data_cluster{i}{1}.frame
            earliest_frame = data_cluster{i}{1}.frame;
        end
    end
end
function [extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, frame)
    intrinsic_params = read_intrinsic(base_path);
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
end
function intrinsic_params = read_intrinsic(base_path)
    txtPath = [base_path 'CameraParams/' 'intrinsics.txt'];
    vec = load(txtPath);
    focal = vec(1); cx = vec(2); cy = vec(3);
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
end
function [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, num_frame, save_path, inter_path] = read_helper_info(helper, ind)
    helper_entry = helper{ind};
    base_path = helper_entry{1};
    GT_Depth_path = helper_entry{2};
    GT_seg_path = helper_entry{3};
    GT_RGB_path = helper_entry{4};
    GT_Color_Label_path = helper_entry{5};
    cam_para_path = helper_entry{6};
    num_frame = helper_entry{7};
    save_path = helper_entry{8};
    inter_path = helper_entry{9};
end
function depth_cluster = init_depth_cluster()
    depth_cluster.depth_maps = cell(0);
    depth_cluster.frame_ind = zeros(0);
end
function data_cluster = read_in_data_cluster(frame, help_entry)
    path = [help_entry{9} 'Instance_map_organized/'];
    mounted = load([path, num2str(frame, '%06d') , '.mat']); data_cluster = mounted.data_cluster;
end
function depth = read_in_depth_map(frame, base_path, GT_Depth_path)
    f = num2str(frame, '%06d');
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
end
function depth_cluster = merge_depth_cluster(depth_map, depth_cluster, frame)
    depth_cluster.depth_maps{end+1} = depth_map;
    depth_cluster.frame_ind(end+1) = frame;
end
function depth_cluster = clean_depth_cluster(data_cluster, depth_cluster)
    earliest_frame = find_earliest_frame_num(data_cluster); selector = false(length(depth_cluster.frame_ind), 1);
    for i = 1 : length(depth_cluster.frame_ind)
        if depth_cluster.frame_ind(i) < earliest_frame
            selector(i) = true;
        end
    end
    depth_cluster.depth_maps(selector) = []; depth_cluster.frame_ind(selector) = [];
end
function save_depth_cluster(frame, depth_cluster, help_entry)
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_organized/';
    path = [help_entry{9} 'Instance_map_organized/'];
    save([path, num2str(frame, '%06d') , '_d.mat'], 'depth_cluster');
end