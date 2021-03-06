function generate_seg_mat(help_info)
    for i = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, num_frame, save_path, inter_path] = read_helper_info(help_info, i);
        seg_sup_mat = generate_sup_mat(num_frame);
        seg_sup_mat = fine_tune_mat(seg_sup_mat, i);
        save_mat(help_info{i}, seg_sup_mat)
    end
end
function seg_sup_mat = fine_tune_mat(seg_sup_mat, ind)
    if ind == 1
        for i = 1 : 76
            seg_sup_mat{i}.min_obj_pixel_num(1) = 800;
            seg_sup_mat{i}.kernel_size(1) = 12;
            seg_sup_mat{i}.map_ration(1) = 0.8;
        end
        for i = 77 : length(seg_sup_mat)
            seg_sup_mat{i}.min_obj_pixel_num(1) = 100;
        end
    end
end
function seg_sup_mat = generate_sup_mat(num_frame)
    min_obj_pixel_num = [200, 400, 200, 20, 20, inf, 200, 200, inf];
    min_height = [0.1, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf];
    max_height = [inf, inf, inf, 1, 4, inf, inf, inf, inf];
    kernel_size = [2, 10, 6, 10, 10, 3, 3, 3, 3];
    map_ration = [1, 0.8, 1, 1, 1, 1, 1, 1, 1];
    seg_sup_mat = cell(num_frame, 1);
    for i = 1 : num_frame
        seg_sup_mat{i}.min_obj_pixel_num = min_obj_pixel_num;
        seg_sup_mat{i}.min_height = min_height;
        seg_sup_mat{i}.max_height = max_height;
        seg_sup_mat{i}.max_height = max_height;
        seg_sup_mat{i}.kernel_size = kernel_size;
        seg_sup_mat{i}.map_ration = map_ration;
    end
end
function save_mat(help_info_entry, seg_sup_mat)
    path = [help_info_entry{9} '/seg_sup_mat.mat'];
    save(path, 'seg_sup_mat');
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