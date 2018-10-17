function union_and_save_all_marks(help_info)
    for jj = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, max_frame, save_path, inter_path] = read_helper_info(helper, jj);
        max_frame = 294;
        for i = 1 : max_frame
            org_entry = read_in_org_entry(i, help_info{jj});
            org_entry = union_single_entry(org_entry);
            save_one_entry(org_entry, i, help_info{jj});
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
function save_one_entry(org_entry, ind, help_info_entry)
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map_unioned/';
    path = [help_info_entry{9} 'seg_unioned_re_sv/'];
    if ind == 1
        mkdir(path);
    end
    save([path num2str(ind, '%06d') '.mat'], 'org_entry')
end
function org_entry = read_in_org_entry(frame, help_info_entry)
    path = [help_info_entry{9} 'seg_raw_re_sv/'];
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/21_Sep_2018_07_segmentation/Instance_map/';
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.prev_mark;
end
function prev_mark = union_single_entry(prev_mark)
    for i = 1 : length(prev_mark)
        to_union_ind = prev_mark{i}.instanceId; union_pool = zeros(length(prev_mark),1); 
        cur_union_pool_count = 1;
        for j = 1 : length(prev_mark)
            if to_union_ind == prev_mark{j}.instanceId
                union_pool(cur_union_pool_count) = j;
                cur_union_pool_count = cur_union_pool_count + 1;
            end
        end
        union_pool = union_pool(union_pool ~=  0);
        prev_mark = union_sub_entry(prev_mark, union_pool);
        if i >= length(prev_mark)
            break;
        end
    end
    ids = get_all_instance_ids(prev_mark);
    if length(ids) ~= length(unique(ids))
        disp('Error');
        prev_mark = zeros(0);
    end
end
function prev_mark = union_sub_entry(prev_mark, union_pool)
    new_entry = prev_mark{union_pool(1)};
    for i = 2 : length(union_pool)
        new_entry = combine_two_entry(new_entry, prev_mark{union_pool(i)});
    end
    prev_mark{union_pool(1)} = new_entry; selector = false(length(prev_mark),1); selector(union_pool(2:end)) = true;
    prev_mark(selector) = [];
end
function entry1 = combine_two_entry(entry1, entry2)
    if entry1.instanceId ~= entry2.instanceId | ~isequal(entry1.color, entry2.color)
        disp('Error'); entry1 = zeors(0);
    end
    entry1.linear_ind = [entry1.linear_ind; entry2.linear_ind];
    entry1.pts_old = [entry1.pts_old; entry2.pts_old];
    entry1.pts_new = [entry1.pts_new; entry2.pts_new];
end

function ids = get_all_instance_ids(prev_mark)
    ids = zeros(length(prev_mark),1);
    for i = 1 : length(prev_mark)
        ids(i) = prev_mark{i}.instanceId;
    end
end