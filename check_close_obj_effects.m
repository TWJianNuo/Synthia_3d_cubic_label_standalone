function check_close_obj_effects(help_info)
    for i = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, max_frame, save_path, inter_path] = read_helper_info(help_info{i});
        for j = 1 : max_frame - 1
           	% data_cluster1 = distill_required_pts(help_info{i}, j);
            % data_cluster2 = distill_required_pts(help_info{i}, j + 1);
            % align_to_previous_frame_and_visualize(data_cluster1, data_cluster2, j, help_info{i});
            process_without_building(help_info{i}, j);
            process_with_building(help_info{i}, j);
        end
    end
end
function process_without_building(help_info_entry, frame)
    excluded_cat = [1 2 3 4 8 10 11 12];
    data_cluster1 = distill_required_pts(help_info_entry, frame,excluded_cat);
    data_cluster2 = distill_required_pts(help_info_entry, frame + 1, excluded_cat);
    [help_info_entry, frame, img1, rgb_new] = align_to_previous_frame_and_visualize(data_cluster1, data_cluster2, frame, help_info_entry);
    save_with_building(help_info_entry, frame, img1, rgb_new);
end
function process_with_building(help_info_entry, frame)
    excluded_cat = [1 3 4 8 10 11 12];
    data_cluster1 = distill_required_pts(help_info_entry, frame,excluded_cat);
    data_cluster2 = distill_required_pts(help_info_entry, frame + 1, excluded_cat);
    [help_info_entry, frame, img1, rgb_new] = align_to_previous_frame_and_visualize(data_cluster1, data_cluster2, frame, help_info_entry);
    save_without_building(help_info_entry, frame, img1, rgb_new)
end
function [help_info_entry, frame, img1, rgb_new] = align_to_previous_frame_and_visualize(data_cluster1, data_cluster2, frame, help_info_entry)
    min_height = 0.05; rgb = data_cluster1.rgb;
    pts_1 = data_cluster1.pts_3d_old; pts_2 = data_cluster2.pts_3d_old;
    adjust_params = data_cluster1.adjust_matrix;
    affine_params = data_cluster1.affine_matrix;
    pts_2_adjusted = (adjust_params * pts_2')';
    pts_1_affined = (affine_params * pts_1')'; selector_1 = pts_1_affined(:,3) > min_height;
    pts_2_affined = (affine_params * pts_2_adjusted')'; selector_2 = pts_2_affined(:,3) > min_height;
    pts_1_affined = pts_1_affined(selector_1, :);
    pts_2_affined = pts_2_affined(selector_2, :);
    [pts_1_2d, depth] = project_point_2d(data_cluster1.extrinsic_params, data_cluster1.intrinsic_params, pts_1_affined, data_cluster1.affine_matrix);
    pts_1_2d = round(pts_1_2d); linear_ind = sub2ind(size(rgb), pts_1_2d(:,2), pts_1_2d(:,1));
    selector_points = false(size(rgb)); selector_points(linear_ind) = true;
    figure('visible', 'off'); clf; scatter3(pts_1_affined(:,1), pts_1_affined(:,2), pts_1_affined(:,3), 3, 'r', 'filled');
    hold on; scatter3(pts_2_affined(:,1), pts_2_affined(:,2), pts_2_affined(:,3), 3, 'g', 'filled');
    axis equal; F = getframe(gcf); [img1, ~] = frame2im(F);
    
    r = rgb(:,:,1); % r = r(:);
    g = rgb(:,:,2); % g = g(:);
    b = rgb(:,:,3); % b = b(:);
    r(selector_points) = 0; g(selector_points) = 255; b(selector_points) = 255;
    rgb_new = uint8(cat(3, r, g, b)); figure('visible', 'off'); clf; imshow(rgb_new);
    % save(help_info_entry, frame, img1, rgb_new)
end
function save_with_building(help_info_entry, frame, img1, img2)
    path = [help_info_entry{9} 'other_obj_align_effects/'];
    if frame == 1
        mkdir(path);
    end
    imwrite(img1, [path num2str(frame), '_3d_wb.png'])
    imwrite(img2, [path num2str(frame), '_2d_wb.png'])
end
function save_without_building(help_info_entry, frame, img1, img2)
    path = [help_info_entry{9} 'other_obj_align_effects/'];
    if frame == 1
        mkdir(path);
    end
    imwrite(img1, [path num2str(frame), '_3d_wotb.png'])
    imwrite(img2, [path num2str(frame), '_2d_wotb.png'])
end
function data_cluster = distill_required_pts(helper_entry, frame, excluded_cat)
    cato = [5, 7, 8, 9, 15]; % excluded_cat = [1 2 3 4 8 10 11 12];
    [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, num_frame, save_path, inter_path] = read_helper_info(helper_entry);
    [extrinsic_params, intrinsic_params, depth_map, label, instance, rgb] = grab_provided_data(base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, frame);
    valid_pts_2d = true(size(label));
    for i = 1 : length(excluded_cat)
        valid_pts_2d = valid_pts_2d & (label ~= excluded_cat(i));
    end
    [yy, xx] = find(valid_pts_2d);
    pts_2d = [xx yy]; linear_ind = sub2ind(size(depth_map), pts_2d(:,2), pts_2d(:,1)); depth_vals = depth_map(linear_ind);
    projected_pts = [pts_2d(:,1) .* depth_vals, pts_2d(:,2) .* depth_vals, depth_vals, ones(length(depth_vals),1)];
    pts_3d_old = (inv(intrinsic_params * extrinsic_params) * projected_pts')';
    % figure(1); clf; scatter3(pts_3d_old(:,1), pts_3d_old(:,2), pts_3d_old(:,3), 3, 'r', 'filled');
    data_cluster.pts_3d_old = pts_3d_old;
    data_cluster.extrinsic_params = extrinsic_params;
    data_cluster.intrinsic_params = intrinsic_params;
    [affine_matrix, adjust_matrix] = read_in_affine_and_adjust(helper_entry, frame);
    data_cluster.affine_matrix = affine_matrix;
    data_cluster.adjust_matrix = adjust_matrix;
    data_cluster.rgb = rgb;
    % r = rgb(:,:,1); % r = r(:);
    % g = rgb(:,:,2); % g = g(:);
    % b = rgb(:,:,3); % b = b(:);
    % r(valid_pts_2d) = 0; g(valid_pts_2d) = 255; b(valid_pts_2d) = 255;
    % rgb_new = uint8(cat(3, r, g, b)); figure(1); clf; imshow(rgb_new)
end
function [affine_matrix, adjust_matrix] = read_in_affine_and_adjust(help_info_entry, frame)
    load([help_info_entry{9} 'affine_matrix.mat'])
    load([help_info_entry{9} 'adjust_matrix.mat'])
    affine_matrix = reshape(affine_matrix_recorder(frame, :), [4 4]);
    adjust_matrix = reshape(param_record(frame, :), [4 4]);
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
function [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, num_frame, save_path, inter_path] = read_helper_info(helper_entry)
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