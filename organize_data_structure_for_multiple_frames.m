% create_rgb_cluster()
% do_multiple_frame_based_optimization();
% check_correctness()
% generate_depth_cluster()
% organize_data_structure_for_multiple_frames();
% org_entry = read_in_org_entry(50); union_single_entry(org_entry)
% [organized_data, depth_map_collection, rgb_collection] = do_preperation(s_frame, e_frame, instance_id);
% cuboid_record = estimate_cubic_shape(organized_data, depth_map_collection);
% visualize_and_save_data(cuboid_record, rgb_collection, organized_data);
% make_dir()
% rectangulars = get_init_guess(mark); optimize_rectangles(rectangulars, depth);
% rectangulars = stack_sampled_pts(rectangulars);
% draw_point_map(rectangulars);
% rectangulars = get_init_guess(mark); rgb = render_image(rgb, rectangulars); 
% draw_point_map(rectangulars); optimize_cubic(rectangulars{1});
% sampled_pts = sample_cubic_by_num(cuboid, sample_pt_num, sample_pt_num); image_size = size(depth_map);
% [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(sampled_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], [image_size(1) image_size(2)], false);
% sampled_pts = sampled_pts(pts_estimated_vlaid, :); pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :); depth = depth(pts_estimated_vlaid);
% camera_origin = (-extrinsic_params(1:3, 1:3)' * extrinsic_params(1:3, 4))';
% cubics = {cuboid}; [visible_pt_3d, ~, ~] = find_visible_pt_global(cubics, pts_estimated_2d, sampled_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
% fin_params = analytical_gradient_v2(obj.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, obj.depth_map, obj.new_pts);
% objs{1}.guess(1:5) = fin_params; cx = objs{1}.guess(1); cy = objs{1}.guess(2); theta = objs{1}.guess(3); l = objs{1}.guess(4); w = objs{1}.guess(5); h = objs{1}.guess(6);
% objs{1}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
function organize_data_structure_for_multiple_frames(help_info)
    for jj = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, max_frame, save_path, inter_path] = read_helper_info(help_info, jj);
        max_hold_frame_num = 10; data_cluster = cell(0);
        % test_correctness(help_info{jj})
        for i = 1 : max_frame
            org_entry = read_in_org_entry(i, help_info{jj});
            data_cluster = add_data_into_cluster(data_cluster, org_entry, i, help_info{jj});
            data_cluster = delete_vanished_entry(data_cluster, i);
            data_cluster = trim_entry_exceeding_max_frame(data_cluster, max_hold_frame_num, help_info{jj}); 
            X = visualize_data_cluster(data_cluster); % figure(1); clf; imshow(X);
            save_all(i, X, data_cluster, help_info{jj});
        end
    end
end
function test_correctness(helper_entry)
    frame1 = 10; frame2 = 20;
    org_entry1 = read_in_org_entry(frame1, helper_entry);
    org_entry2 = read_in_org_entry(frame2, helper_entry);
    frame_tune_matrix = calculate_transformation_matrix(frame1, frame2, helper_entry);
    figure(1); clf;
    for i = 1 : length(org_entry1)
        pts_new = org_entry1{i}.pts_new;
        scatter3(pts_new(:,1), pts_new(:,2), pts_new(:,3), 3, 'r', 'fill'); hold on;
    end
    for i = 1 : length(org_entry2)
        pts_old = org_entry2{i}.pts_old;
        pts_old = (frame_tune_matrix * pts_old')';
        affine = org_entry1{1}.affine_matrx;
        pts_new = (affine * pts_old')';
        scatter3(pts_new(:,1), pts_new(:,2), pts_new(:,3), 3, 'g', 'fill'); hold on;
    end
end
function X = visualize_data_cluster(data_cluster)
    sample_rate = 10;
    f = figure('visible', 'off'); clf;
    for i = 1 : length(data_cluster)
        for j = 1 : length(data_cluster{i})
            pts = data_cluster{i}{j}.pts_new;
            pts = pts(1:sample_rate:end, :);
            color = data_cluster{i}{j}.color;
            scatter3(pts(:,1), pts(:,2), pts(:,3), 3, color, 'fill'); hold on;
        end
    end
    axis equal; F = getframe(gcf); [X, ~] = frame2im(F);
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
function save_all(frame, X, data_cluster, help_info_entry)
    path1 = [help_info_entry{9} '/Instance_map_organized/'];
    path2 = [help_info_entry{9} '/Instance_map_organized_visualize/'];
    if frame == 1
        mkdir(path1); mkdir(path2);
    end
    save([path1 num2str(frame, '%06d') '.mat'], 'data_cluster');
    imwrite(X, [path2 num2str(frame, '%06d') '.png']);
end
function data_cluster = trim_entry_exceeding_max_frame(data_cluster, max_frame_num, help_entryy)
    for i = 1 : length(data_cluster)
        if length(data_cluster{i}) > max_frame_num
            [data_cluster{i}, restore_matrix] = back_prop_one_frame(data_cluster{i}, help_entryy);
            data_cluster{i}(1) = [];
            data_cluster{i}{1}.restore_matrix = restore_matrix;
        else
            idt_mx = eye(4);
            data_cluster{i}{1}.restore_matrix = idt_mx;
        end
    end
end
function [entry, restore_matrix] = back_prop_one_frame(entry, help_entryy)
    load([help_entryy{9} 'adjust_matrix.mat'])
    cur_frame = entry{1}.frame;
    restore_matrix = inv(reshape(param_record(cur_frame, :), [4,4])); 
    for i = 2 : length(entry)
        entry{i}.pts_old = (restore_matrix * entry{i}.pts_old')';
        entry{i}.pts_new = (entry{2}.affine_matrx * entry{i}.pts_old')';
        entry{i}.extrinsic_params = entry{i}.extrinsic_params * reshape(param_record(cur_frame, :), [4,4]);
    end
    restore_matrix = restore_matrix * inv(entry{1}.affine_matrx);
end
function data_cluster = add_data_into_cluster(data_cluster, cur_entry, frame, helper_entry)
    for i = 1 : length(cur_entry)
        obj = cur_entry{i};
        position_ind = find_pos_in_data_cluster(data_cluster, obj);
        if position_ind == 0
            data_cluster = concatinate_data_entry_to_cluster(data_cluster, {obj});
        else
            data_cluster = add_data_entry(data_cluster, obj, position_ind, frame, helper_entry);
        end
    end
end
function data_cluster = delete_vanished_entry(data_cluster, frame_num)
    selector = false(length(data_cluster), 1);
    for i = 1 : length(data_cluster)
        if data_cluster{i}{end}.frame ~= frame_num
            selector(i) = true;
        end
    end
    data_cluster(selector) = [];
end
function data_cluster = add_data_entry(data_cluster, to_add_obj, ind, frame_end, helper_entry)
    frame_first = data_cluster{ind}{1}.frame;
    frame_tune_matrix = calculate_transformation_matrix(frame_first, frame_end, helper_entry);
    to_add_obj = adjust_the_obj(to_add_obj, data_cluster{ind}{1}, frame_tune_matrix);
    data_cluster{ind}{end+1,1} = to_add_obj;
end
function data_cluster = concatinate_data_entry_to_cluster(data_cluster, cur_entry)
    data_cluster{end+1, 1} = cur_entry;
end
function position_ind = find_pos_in_data_cluster(data_cluster, obj)
    position_ind = 0; 
    if isempty(data_cluster)
        position_ind = 0;
    end
    for i = 1 : length(data_cluster)
        if data_cluster{i}{1}.instanceId == obj.instanceId
            position_ind = i;
        end
    end
end
function org_entry = read_in_org_entry(frame, help_info_entry)
    path = [help_info_entry{9} '/seg_unioned_re_sv/'];
    ind = num2str(frame, '%06d');
    loaded = load([path ind '.mat']); org_entry = loaded.org_entry;
end
function frame_tune_matrix = calculate_transformation_matrix(start_frame, end_frame, helper_entry)
    load([helper_entry{9} 'adjust_matrix.mat'])
    frame_tune_matrix = eye(4);
    for i = start_frame : end_frame - 1
        frame_tune_matrix = frame_tune_matrix * reshape(param_record(i, :), [4 4]);
    end
end
function the_obj = adjust_the_obj(the_obj, org_obj, frame_tune_matrix)
    the_obj.extrinsic_params = the_obj.extrinsic_params * inv(frame_tune_matrix);
    the_obj.pts_old = (frame_tune_matrix * the_obj.pts_old')';
    the_obj.pts_new = (org_obj.affine_matrx * the_obj.pts_old')';
end
function visualize_between_two_obj(the_obj, org_obj)
    pts1 = the_obj.pts_new;
    pts2 = org_obj.pts_new;
    figure(1); clf; scatter3(pts1(:,1), pts1(:,2), pts1(:,3), 3, 'r', 'fill'); hold on;
    scatter3(pts2(:,1), pts2(:,2), pts2(:,3), 3, 'g', 'fill'); hold on;
end