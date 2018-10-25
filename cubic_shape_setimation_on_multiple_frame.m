function cubic_shape_setimation_on_multiple_frame(help_info)
    % load('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Synthia_3D_scenen_reconstruction_standalone/output_results/SYNTHIA-SEQS-05-SPRING/24_Oct_2018_15_mul/3_70.mat');
    % cubic_record_entry = optimize_for_single_obj_set(cubic_record_entry, objs, depth_cluster, frame_num, obj_ind);
    %{
    rgb = grab_rgb_by_mat(1, help_info{1});
    data_cluster = {objs}; cubic_cluster = {cubic_record_entry};
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    for i = 1 : length(data_cluster)
        color = uint8(ceil(data_cluster{i}{end}.color * 255));
        r(data_cluster{i}{end}.linear_ind) = color(1);
        g(data_cluster{i}{end}.linear_ind) = color(2);
        b(data_cluster{i}{end}.linear_ind) = color(3);
    end
    rgb_ = cat(3, r, g, b);
    for i = 1 : length(data_cluster)
        intrinsic_params = data_cluster{i}{end}.intrinsic_params;
        extrinsic_params = data_cluster{i}{end}.extrinsic_params * inv(data_cluster{i}{1}.affine_matrx);
        rgb__ = cubic_lines_of_2d(rgb_, cubic_cluster{i}.cuboid, intrinsic_params, extrinsic_params);
    end
    color = uint8(ceil(rand(1,3) * 255));
    pts = objs{1}.pts_new; pts_2d = project_point_2d(extrinsic_params, intrinsic_params, pts);
    pts_2d = round(pts_2d); linear_ind_new = sub2ind([size(rgb,1), size(rgb,2)], pts_2d(:,2), pts_2d(:,1));
    r(linear_ind_new) = color(1);
    g(linear_ind_new) = color(2);
    b(linear_ind_new) = color(3);
    rgb_ = cat(3, r, g, b);
    % img = cubic_lines_of_2d(rgb, cubic_record_entry.cuboid, objs{1}.intrinsic_params, objs{1}.extrinsic_params * inv(objs{1}.affine_matrx));
    figure(1); clf; imshow(rgb__)
    figure(2); clf; draw_cubic_shape_frame(cubic_record_entry.cuboid); 
    hold on;  scatter3(pts(:,1),pts(:,2),pts(:,3),3,'g','fill'); axis equal
    %}
    global path_mul
    for jj = 1 : length(help_info)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, max_frame, save_path, inter_path] = read_helper_info(help_info, jj);
        cubic_cluster = zeros(0); make_dir(help_info{jj});
        for frame = 1 : 5
            save([path_mul num2str(frame) '.mat'])
            % Prepare data
            rgb = grab_rgb_by_mat(frame, help_info{jj});
            [data_cluster, depth_cluster] = read_in_clusters(frame, help_info{jj});
            data_cluster = exclude_data_cluster_of_moving_things(data_cluster);
            
            % [data_cluster, cubic_cluster] = optimize_cubic_shape_for_data_cluster(data_cluster, depth_cluster, cubic_cluster, frame);
            
            cubic_cluster = get_cubic_cluster(data_cluster, cubic_cluster);
            cubic_cluster = restore_changed_cubics(cubic_cluster, data_cluster);
            [data_cluster, cubic_cluster] = trim_incontinuous_frame(data_cluster, cubic_cluster);
            for i = 1 : length(data_cluster)
                cubic_cluster{i} = optimize_for_single_obj_set(cubic_cluster{i}, data_cluster{i}, depth_cluster, frame, i);
            end
            
            % Visualize results and save
            rgb = render_image(rgb, cubic_cluster, data_cluster);
            metric_record = 1; save_results(metric_record, rgb, frame);
            % render_in_3d(data_cluster, cubic_cluster, frame);
            % draw_and_check_r1esults(data_cluster, cubic_cluster, frame)
        end
    end
end
function check_data_cluster(objs, depth_cluster)
    % figure(1); clf; 
    sz_depth_map = size(depth_cluster.depth_maps{1});
    for i = 1 : length(objs)
        for j = 1 : length(objs{i})
            z_mat = inv(objs{i}{j}.intrinsic_params * (objs{i}{j}.extrinsic_params * inv(objs{i}{j}.affine_matrx)));
            depth_map_ind = objs{i}{j}.frame - depth_cluster.frame_ind(1) + 1;
            [yy, xx] = ind2sub(sz_depth_map, objs{i}{j}.linear_ind); pixel_loc = [xx yy];
            depth_inv = depth_cluster.depth_maps{depth_map_ind}(objs{i}{j}.linear_ind);
            x_inv_gt = (z_mat * ([xx .* depth_inv, yy .* depth_inv, depth_inv ones(length(depth_inv), 1)])')';
            sum(sum((x_inv_gt(:,1:3) - objs{i}{j}.pts_new(:,1:3)).^2))
            % hold on; scatter3(x_inv_gt(:,1),x_inv_gt(:,2),x_inv_gt(:,3),3,'g','fill');
            % hold on; scatter3(objs{i}{j}.pts_new(:,1),objs{i}{j}.pts_new(:,2),objs{i}{j}.pts_new(:,3),3,'b','fill')
        end
    end
end
function render_in_3d(data_cluster, cubic_cluster, frame)
    global path_mul
    figure(1); clf; 
    for i = 1 : length(data_cluster)
        draw_cubic_shape_frame(cubic_cluster{i}.cuboid); hold on;
        for j = 1 : length(data_cluster{i})
            color = data_cluster{i}{j}.color;
            scatter3(data_cluster{i}{j}.pts_new(:,1), data_cluster{i}{j}.pts_new(:,2), data_cluster{i}{j}.pts_new(:,3), 3, color, 'fill');
        end
    end
    savefig([path_mul 'rgb_image/' num2str(frame) '.fig'])
end
function data_cluster = exclude_data_cluster_of_moving_things(data_cluster)
    selector = false(size(data_cluster,1), 1);
    for i = 1 : length(selector)
        if data_cluster{i}{1}.instanceId > 10000
            selector(i) = true;
        end
    end
    data_cluster(selector) = [];
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
function rgb = grab_rgb_by_mat(frame, help_info_entry)
    local_path = [help_info_entry{9} 'rgb_cluster/'];
    load([local_path num2str(frame) '.mat']);
end
function save_results(metric_record, rgb, frame)
    global path_mul
    imwrite(rgb, [path_mul 'rgb_image/' num2str(frame), '.png']);
    if frame == 1
        f1 = fopen([path_mul 'rgb_image/' 'metric.txt'],'w');
    else
        f1 = fopen([path_mul 'rgb_image/' 'metric.txt'],'a');
    end
    print_matrix(f1, frame);
    print_matrix(f1, metric_record);
end
function rgb = render_image(rgb, cubic_cluster, data_cluster)
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    for i = 1 : length(data_cluster)
        color = uint8(ceil(data_cluster{i}{end}.color * 255));
        r(data_cluster{i}{end}.linear_ind) = color(1);
        g(data_cluster{i}{end}.linear_ind) = color(2);
        b(data_cluster{i}{end}.linear_ind) = color(3);
    end
    rgb = cat(3, r, g, b);
    for i = 1 : length(data_cluster)
        intrinsic_params = data_cluster{i}{end}.intrinsic_params;
        extrinsic_params = data_cluster{i}{end}.extrinsic_params * inv(data_cluster{i}{1}.affine_matrx);
        rgb = cubic_lines_of_2d(rgb, cubic_cluster{i}.cuboid, intrinsic_params, extrinsic_params);
    end
end
function [data_cluster, cubic_cluster] = optimize_cubic_shape_for_data_cluster(data_cluster, depth_cluster, cubic_cluster, frame)
end
function ave_dist = metric2_single(cuboid, objs, depth_map_cluster)
    tot_dist = 0; tot_num = 0;
    for i = 1 : length(objs)
        depth_map = distill_depth_map_frame_depth_cluster(objs{i}, depth_map_cluster);
        pts_3d = get_all_3d_pts(objs{i}, depth_map, objs{1}.affine_matrx);
        [dist, num] = get_dist_and_num_record(cuboid, pts_3d);
        tot_dist = tot_dist + dist; tot_num = tot_num + num;
    end
    ave_dist = tot_dist / tot_num;
end
function [tot_dist, num] = get_dist_and_num_record(cuboid, pts_3d)
    [~, dist_record] = calculate_ave_distance(cuboid, pts_3d);
    num = length(dist_record); tot_dist = sum(dist_record);
end
function [loss, num_on_cubic] = metric_pos_and_inv(cuboid, objs, depth_cluster)
    sample_num = 10; loss = 0; num_on_cubic = 0;
    sampled_pts = sample_cubic_by_num(cuboid, sample_num, sample_num);
    sz_depth_map = size(depth_cluster.depth_maps{1});
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    plane_param1 = cuboid{1}.params; plane_param2 = cuboid{2}.params;
    [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h);
    for i = 1 : length(objs)
        z_mat = inv(objs{i}.intrinsic_params * (objs{i}.extrinsic_params * inv(objs{i}.affine_matrx)));
        % Find visible points approximately
        visible_pts_ind = false(size(sampled_pts,1),1);
        visible_plane_ind = find_visible_plane(cuboid, objs{i}.intrinsic_params, objs{i}.extrinsic_params * inv(objs{1}.affine_matrx));
        for j = 1 : length(visible_plane_ind)
            visible_pts_ind = visible_pts_ind | (sampled_pts(:,4) == visible_plane_ind(j));
        end
        visible_pts = sampled_pts(visible_pts_ind, :);
        % Get depth map ind
        depth_map_ind = objs{i}.frame - depth_cluster.frame_ind(1) + 1;
        % Calculate metric pos
        [pts2d_pos, depth_pos] = project_point_2d(objs{i}.extrinsic_params, objs{i}.intrinsic_params, visible_pts(:, 1:3), objs{1}.affine_matrx);
        pts2d_pos = round(pts2d_pos);
        selector_pos = (pts2d_pos(:,1) > 0) & (pts2d_pos(:,1) <= sz_depth_map(2)) & (pts2d_pos(:,2) > 0) & (pts2d_pos(:,2) < sz_depth_map(1));
        linear_ind_pos = sub2ind(sz_depth_map, pts2d_pos(selector_pos,2), pts2d_pos(selector_pos,1)); ica = ismember(linear_ind_pos, objs{i}.linear_ind);
        selector_pos(selector_pos) = ica;
        loss_pos = sum(abs(depth_cluster.depth_maps{depth_map_ind}(linear_ind_pos(ica)) - depth_pos(selector_pos))) / length(linear_ind_pos);
        % ratio_pos = sum(selector_pos) / length(selector_pos);
        % Calculate metric inv
        [yy, xx] = ind2sub(sz_depth_map, objs{i}.linear_ind); pixel_loc = [xx yy];
        depth_inv = depth_cluster.depth_maps{depth_map_ind}(objs{i}.linear_ind);
        [~, plane_type_rec, selector_inv, x11, x22, d1, d2] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, objs{i}.extrinsic_params * inv(objs{1}.affine_matrx));
        select_p1 = selector_inv; select_p1(select_p1) = ((x11(:,1) >= 0) & (plane_type_rec == 1));
        select_p2 = selector_inv; select_p2(select_p2) = ((x22(:,1) >= 0) & (plane_type_rec == 2));
        loss_inv = sum(abs(depth_inv(select_p1) - d1((x11(:,1) >= 0) & (plane_type_rec == 1)))) + ...
                   sum(abs(depth_inv(select_p2) - d2((x22(:,1) >= 0) & (plane_type_rec == 2))));
        loss_inv = loss_inv / (sum(select_p1) + sum(select_p2));
        % ratio_inv = (sum(select_p1) + sum(select_p2)) / length(select_p1);
        num_on_cubic = num_on_cubic + sum(select_p1) + sum(select_p2);
        loss = loss + (loss_inv + loss_pos) / 2;
    end
    loss = loss / length(objs);
end
function best_cuboid_entry = optimize_for_single_obj_set(cubic_record_entry, objs, depth_cluster, frame_num, obj_ind)
    global path_mul
    % Make sure all the points are near the surface
    % Cut the cubic shape after the optimization
    % save([path_mul num2str(frame_num) '_' num2str(obj_ind) '.mat'])
    % load('/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Synthia_3D_scenen_reconstruction_standalone/output_results/SYNTHIA-SEQS-05-SPRING/18_Oct_2018_23_mul/1_70.mat');
    activation_label = cubic_record_entry.activation_label; depth_cluster_ = image_blur(depth_cluster);
    it_num = 200; loss_record = zeros(it_num, 1); 
    org_cubic_record_entry = cubic_record_entry;
    delta_record_norm = zeros(it_num, 1); num_on_pt_th = 0.15; min_loss = inf;
    [loss_org, num_on_cubic_org] = metric_pos_and_inv(org_cubic_record_entry.cuboid, objs, depth_cluster);
    [counts_set, linear_ind_set, inv_selector_set, isvalid] = get_useable_sample_for_obj_cluster(objs, depth_cluster_, cubic_record_entry.visible_pts, cubic_record_entry.cuboid);
    if ~isvalid
        best_cuboid_entry = truncate_cuboid(objs, depth_cluster, org_cubic_record_entry, activation_label, true); 
        % save_visualize(best_cuboid_entry, objs, 0, frame_num, obj_ind, inv_selector_set, counts_set);
        return
    end
    for i = 1 : it_num
        cuboid = cubic_record_entry.cuboid; visible_pts = cubic_record_entry.visible_pts;
        diff_sum = 0; hess_sum = 0; loss_sum = 0;
        delta_norm_record = zeros(length(objs),1);
        for j = 1 : length(objs)
            if (sum(counts_set(j,:)) == 0) || (sum(linear_ind_set{j}) == 0)
                continue;
            end
            depth_map = grab_depth_map(depth_cluster_, objs{j});
            
            [diff, hess, loss] = accum_for_one_obj(cuboid, objs{j}, depth_map, visible_pts(counts_set(j,:), :), activation_label, linear_ind_set{j});
            % visualize_inner_sit(cuboid, objs{j}, depth_map, visible_pts(counts_set(j,:), :), activation_label, linear_ind_set{j});
            diff_sum = diff_sum + diff; hess_sum = hess_sum + hess; loss_sum = loss_sum + loss;
            delta_norm_record(j) = norm(get_delta_from_diff_and_hess(diff, hess));
        end
        % save_visualize(best_cuboid_entry, objs, i, frame_num, obj_ind, inv_selector_set);
        delta_theta = get_delta_from_diff_and_hess(diff_sum, hess_sum);
        loss_record(i) = loss_sum;
        delta_record_norm(i) = norm(delta_theta);
        
        % Find cubic shape with minimum loss
        if min_loss > loss_sum
            min_loss = loss_sum;
            best_cuboid_entry = cubic_record_entry;
        end
        
        cubic_record_entry = update_cuboid_entry(cubic_record_entry, delta_theta, activation_label);
        if judge_stop(delta_theta, cubic_record_entry.cuboid, loss_record, delta_record_norm)
            % break;
        end
    end
    best_cuboid_entry = truncate_cuboid(objs, depth_cluster, best_cuboid_entry, activation_label, false);
    [loss_cur, num_on_cubic_cur] = metric_pos_and_inv(best_cuboid_entry.cuboid, objs, depth_cluster);
    % figure(2); clf; stem(loss_record(loss_record~=0))
    if (num_on_cubic_cur < num_on_cubic_org *(1 - num_on_pt_th)) || (loss_org < loss_cur)
        best_cuboid_entry = truncate_cuboid(objs, depth_cluster, org_cubic_record_entry, activation_label, true);
        % save_visualize(best_cuboid_entry, objs, it_num, frame_num, obj_ind, inv_selector_set, counts_set);
    else
        % save_visualize(best_cuboid_entry, objs, it_num, frame_num, obj_ind, inv_selector_set, counts_set);
        % save_stem(loss_record(loss_record ~= 0), frame_num, obj_ind);
    end
end
function depth_map_cluster = image_blur(depth_map_cluster)
    for i = 1 : length(depth_map_cluster.depth_maps)
        depth_map_cluster.depth_maps{i} = imgaussfilt(depth_map_cluster.depth_maps{i},'FilterSize',3);
    end
end
function metric_record = calculate_metric(cuboid_cluster, objs_cluster, depth_map_cluster)
    metric_record = zeros(length(cuboid_cluster), 2);
    for i = 1 : length(cuboid_cluster)
        metric_record(i,1) = metric1_single(cuboid_cluster{i}.cuboid, objs_cluster{i}, depth_map_cluster);
        metric_record(i,2) = metric2_single(cuboid_cluster{i}.cuboid, objs_cluster{i}, depth_map_cluster);
    end
end
function pts_3d = get_all_3d_pts(obj, depth_map, affine)
    linear_ind = obj.linear_ind; extrinsic = obj.extrinsic_params * inv(affine); intrinsic = obj.intrinsic_params;
    depth_val = depth_map(linear_ind); sz_depth = size(depth_map);
    [yy, xx] = ind2sub([sz_depth(1) sz_depth(2)], linear_ind);
    pts_transed = [xx .* depth_val, yy .* depth_val, depth_val, ones(size(depth_val,1),1)];
    pts_3d = (inv(intrinsic * extrinsic) * pts_transed')';
end
function [visible_plane_ind, visible_pt_label] = find_visible_plane(cuboid, intrinsic_params, extrinsic_params)
    params = generate_cubic_params(cuboid); visible_plane_ind = 1 : 4;
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    c1 = [
        xc + 1 / 2 * w * sin(theta);
        yc - 1 / 2 * w * cos(theta);
        h/2;
        1;
        ];
    c2 = [
        xc + 1/2 * l * cos(theta);
        yc + 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    c3 = [
        xc - 1/2 * w * sin(theta);
        yc + 1/2 * w * cos(theta);
        1/2 * h;
        1
        ];
    c4 = [
        xc - 1/2 * l * cos(theta);
        yc - 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    visible_pt_label = find_visible_pt_global_({cuboid}, [c1';c2';c3';c4'], intrinsic_params, extrinsic_params);
    % visible_pt_label = find_visible_pt_global({cuboid}, [c1';c2';c3';c4'], intrinsic_params, extrinsic_params);
    visible_plane_ind = visible_plane_ind(visible_pt_label);
end

function save_stem(loss_record, frame_num, obj_ind)
    global path_mul
    loss_record = loss_record(loss_record~=0);
    figure('visible', 'off'); clf; stem(loss_record', 'filled');
    F = getframe(gcf); [X, ~] = frame2im(F);
    % path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/cubic_shape_estimation/single_frame_exp/';
    imwrite(X, [path_mul num2str(frame_num) '_' num2str(obj_ind) '_loss' '.png']);
end
function save_visualize(cubic_record_entry, objs, it_num, frame_num, obj_ind, inv_selector_set, counts_set)
    global path_mul
    if mod(it_num, 0) ~= -1
        % figure('visible', 'off'); clf; draw_cubic_shape_frame(cubic_record_entry.cuboid); hold on;
        figure(1); clf; draw_cubic_shape_frame(cubic_record_entry.cuboid); hold on;
        pts_cubic = cubic_record_entry.visible_pts;
        
        for i = 1 : length(objs)
            valid_3d = counts_set(i,:);
            scatter3(pts_cubic(valid_3d,1), pts_cubic(valid_3d,2), pts_cubic(valid_3d,3), 3, 'r', 'fill'); hold on;
            pts = objs{i}.pts_new; selector = inv_selector_set{i};
            scatter3(pts(:,1), pts(:,2), pts(:,3), 3, 'g', 'fill'); hold on;
            scatter3(pts(selector,1), pts(selector,2), pts(selector,3), 5, 'b', 'fill'); hold on;
        end
        axis equal; F = getframe(gcf); [X, ~] = frame2im(F);
        imwrite(X, [path_mul num2str(frame_num) '_' num2str(obj_ind) '_' num2str(it_num) '.png']); pause(1)
    end
end
function to_stop = judge_stop(delta, cuboid, diff_record, delta_record_norm)
    params = generate_cubic_params(cuboid); delta_record_norm = delta_record_norm(delta_record_norm~=0); delta_norm_th = 0.00001;
    th = 0.0001; to_stop = false; diff_record = diff_record(diff_record~=0); step_range = 10; th_hold = 0.1;
    if max(abs(delta)) < th || params(4) < 0 || params(5) < 0
        to_stop = true;
    end
    if length(diff_record) > step_range
        if abs(diff_record(end) - diff_record(end - step_range)) < th_hold
            to_stop = true;
        end
    end
    if sum(delta_record_norm < delta_norm_th) > 10
        to_stop = true;
    end
end
function cubic_record = update_cuboid_entry(cubic_record, delta_theta, activation_label)
    cuboid = cubic_record.cuboid; params = generate_cubic_params(cuboid); activation_label = (activation_label == 1);
    params(activation_label) = params(activation_label) + delta_theta;
    n_cuboid = generate_center_cuboid_by_params(params);
    cubic_record.cuboid = n_cuboid; % pts_bck = cubic_record.visible_pts;
    new_pt = pts_3d(cubic_record.visible_pts(:, 5:6), cubic_record.visible_pts(:, 4), params(1), params(2), params(3), params(4), params(5), params(6))';
    cubic_record.visible_pts(:, 1:3) = new_pt(:,1:3);
    % figure(1); clf; draw_cubic_shape_frame(cuboid)
    % hold on; scatter3(pts_bck(:,1),pts_bck(:,2),pts_bck(:,3),3,'b','fill')
    % hold on; scatter3(cubic_record.visible_pts(:,1),cubic_record.visible_pts(:,2),cubic_record.visible_pts(:,3),3,'g','fill')
end
function pts3 = pts_3d(k, plane_ind, theta, xc, yc, l, w, h)
    pts3 = zeros(4, length(plane_ind));
    for cur_plane_ind = 1 : 4
        selector = (plane_ind == cur_plane_ind);
        k1 = k(selector,1)'; k2 = k(selector,2)';
        if cur_plane_ind == 1
            pts3(:, selector) = [
                xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * cos(theta) * l;
                yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * sin(theta) * l;
                k2 * h;
                ones(size(k1));
                ];
        end
        if cur_plane_ind == 2
            pts3(:, selector) = [
                xc + 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
                yc + 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
                k2 * h;
                ones(size(k1));
                ];
        end
        if cur_plane_ind == 3
            pts3(:, selector) = [
                xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
                yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
                k2 * h;
                ones(size(k1));
                ];
        end
        if cur_plane_ind == 4
            pts3(:, selector) = [
                xc - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
                yc - 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
                k2 * h;
                ones(size(k1));
                ];
        end
    end
end
function delta = get_delta_from_diff_and_hess(sum_diff, sum_hess)
    if sum(isnan(sum_diff)) + sum(sum(isnan(sum_hess))) > 0
        delta = 0;
    else
        delta = sum_diff * inv(sum_hess);
    end
end
function depth_map = grab_depth_map(depth_cluster, cur_obj)
    frame_ind = cur_obj.frame;
    ind = find(depth_cluster.frame_ind == frame_ind); depth_map = depth_cluster.depth_maps{ind};
end
function visualize_inner_sit(cuboid, cur_obj, depth_map, visible_pts, activation_label, linear_ind)
    % pts_3d = get_all_3d_pts(cur_obj, depth_map, cur_obj.affine_matrx);
    visible_pts = [visible_pts(:, 5) visible_pts(:, 6) visible_pts(:, 4)];
    extrinsic_param = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic_param = cur_obj.intrinsic_params;
    color1 = rand(size(linear_ind,1),3); color2 = rand(size(visible_pts,1),3);
    
    [~, ~, ~, dominate_pts, dominate_color] = multiple_frame_cubic_estimation(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label, color1, color2);
    % [depth_map_projected] = map_pts_to_depth_map(cur_obj.extrinsic_params, cur_obj.intrinsic_params, cur_obj.affine_matrx, dominate_pts, depth_map, dominate_color);
    % hold on; scatter3(pts_3d(:,1), pts_3d(:,2), pts_3d(:,3), 4, 'k', 'fill');
    % figure(2); clf; imshow(depth_map_projected)
end
function [depth_map, pts_2d] = map_pts_to_depth_map(extrinsic, intrinsic, affine, pts, depth_map, colors)
    [pts_2d, depth_vals] = project_point_2d(extrinsic, intrinsic, pts, affine); pts_2d = round(pts_2d);
    depth_map = mark_on_depth_map(depth_map, pts_2d, colors);
end
function depth_map = mark_on_depth_map(depth_map, pos, color)
    depth_map = uint8(255 * (depth_map / max(max(depth_map))));
    depth_map = cat(3,depth_map,depth_map,depth_map);
    depth_map = insertShape(depth_map,'FilledCircle',[pos(:,1) pos(:,2) ones(length(pos(:,1)),1) * 1],'LineWidth',1,'Color',uint8(255 * color));
end
function [diff, hess, loss] = accum_for_one_obj(cuboid, cur_obj, depth_map, visible_pt_3d, activation_label, linear_ind)
    % tot_pos_num = size(visible_pt_3d, 1); linear_ind = cur_obj.linear_ind; tot_inv_num = size(linear_ind, 1);
    % linear_ind = down_sample_linear_ind(linear_ind, tot_pos_num);
    visible_pt_3d = [visible_pt_3d(:, 5) visible_pt_3d(:, 6) visible_pt_3d(:, 4)];
    extrinsic_param = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic_param = cur_obj.intrinsic_params;
    % [diff_, hess_, loss_] = analytical_gradient_combined_v2_mult_frame(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pt_3d, activation_label);
    [diff, hess, loss] = multiple_frame_cubic_estimation(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pt_3d, activation_label);
    % visualize_combine_multi(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pts, activation_label);
end
function is_visible_record = get_visible_objs(cuboid, objs)
    is_visible_record = true(size(objs,1),1);
    for i = 1 : length(objs)
        is_visible_record(i) = judge_is_still_visible(cuboid, objs{i});
    end
end
function [depth_map, pos_ind] = distill_depth_map_frame_depth_cluster(cur_obj, depth_cluster)
    frame_num = cur_obj.frame; pos_ind = 0;
    for i = 1 : length(depth_cluster.frame_ind)
        if frame_num == depth_cluster.frame_ind(i)
            pos_ind = i;
        end
    end
    depth_map = depth_cluster.depth_maps{pos_ind};
end
function cuboid_entry = truncate_cuboid(objs, depth_cluster, cuboid_entry, activation_label, is_org)
    cuboid = cuboid_entry.cuboid;
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    plane_param1 = cuboid{1}.params; plane_param2 = cuboid{2}.params;
    [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h);
    border_ratio = 0.97; border_ratio_h = 0.98; border_ratio_minus_h = 0.01; 
    limit_l_re = zeros(length(objs), 1); limit_w_re = zeros(length(objs), 1); limit_h_re = zeros(length(objs), 1); limit_mh_re = zeros(length(objs), 1);
    sz_depth_map = size(depth_cluster.depth_maps{1});
    for i = 1 : length(objs)
        z_mat = inv(objs{i}.intrinsic_params * (objs{i}.extrinsic_params * inv(objs{i}.affine_matrx)));
        depth_map_ind = objs{i}.frame - depth_cluster.frame_ind(1) + 1;
        [yy, xx] = ind2sub(sz_depth_map, objs{i}.linear_ind); pixel_loc = [xx yy];
        depth_inv = depth_cluster.depth_maps{depth_map_ind}(objs{i}.linear_ind);
        x_inv_gt_ = (z_mat * ([xx .* depth_inv, yy .* depth_inv, depth_inv ones(length(depth_inv), 1)])')';
        [~, plane_type_rec, selector_inv] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, (objs{i}.extrinsic_params * inv(objs{i}.affine_matrx)));
        x_inv_gt = x_inv_gt_(selector_inv, :);
        
        selector1 = (plane_type_rec == 1);
        selector2 = (plane_type_rec == 2);
        xx1 = (tans_sign_mat1 * x_inv_gt(selector1, :)')';
        xx2 = (tans_sign_mat2 * x_inv_gt(selector2, :)')';
        
        hh = sort(x_inv_gt_(:,3)); xx1 = sort(xx1(:,1)); xx2 = sort(xx2(:,1)); 
        try
            limit_l_re(i) = xx1(floor(length(xx1) * border_ratio));
        catch
            limit_l_re(i) = l;
        end
        try
            limit_w_re(i) = xx2(floor(length(xx2) * border_ratio));
        catch
            limit_w_re(i) = w;
        end
        try
            limit_h_re(i) = hh(floor(length(hh) * border_ratio_h));
        catch
            limit_h_re(i) = h;
        end
        try
            limit_mh_re(i) = hh(floor(length(hh) * border_ratio_minus_h));
            if limit_mh_re(i) > 0
                limit_mh_re(i) = 0;
            end
        catch
            limit_mh_re(i) = 0;
        end
    end
    limit_l = max(limit_l_re); limit_w = max(limit_w_re); limit_h = max(limit_h_re); limit_mh = min(limit_mh_re);
    center = [xc, yc];
    center = center + (limit_l - l) / 2 * [-cos(theta), -sin(theta)] + (limit_w - w) / 2 * [-sin(theta), cos(theta)];
    xc_ = center(1); yc_ = center(2); l_ = limit_l; w_ = limit_w; h = limit_h;
    if is_org
        params = [theta, xc, yc, l, w, h];
    else
        params = [theta, xc_, yc_, l_, w_, h];
    end
    cuboid_entry.cuboid = generate_center_cuboid_by_params(params); delta_theta = zeros(1,sum(activation_label));
    cuboid_entry = update_cuboid_entry(cuboid_entry, delta_theta, activation_label);
    cuboid_entry.cuboid{1}.mh = limit_mh;
    
    % figure(1); draw_cubic_shape_frame(cuboid); hold on; draw_cubic_shape_frame(cuboid_entry.cuboid)
    % figure(1); clf; draw_cubic_shape_frame(cuboid);
    % hold on; scatter3(x_inv_gt(selector1, 1),x_inv_gt(selector1, 2),x_inv_gt(selector1, 3),3,'g','fill')
    % hold on; scatter3(x_inv_gt(selector2, 1),x_inv_gt(selector2, 2),x_inv_gt(selector2, 3),3,'b','fill')
    % axis equal
end
function [counts_set, selected_linear_ind, linaer_ind_selector, isvalid] = get_useable_sample_for_obj_cluster(objs, depth_cluster, visible_pts, cuboid)
    % Distill points that are approximately to be true
    counts_set = false(size(objs, 1), size(visible_pts,1)); sz_depth_map = size(depth_cluster.depth_maps{1});
    grad_th = 100; plane_th = 0.25;
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    plane_param1 = cuboid{1}.params; plane_param2 = cuboid{2}.params;
    [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h);
    selected_linear_ind = cell(length(objs), 1); linaer_ind_selector = cell(length(objs), 1);
    max_pt_p1 = zeros(length(objs),4); max_pt_p2 = zeros(length(objs),4); area_th = 0.01; isvalid = true;
    tot_valid_pt_num = 0; tot_valid_pt_num_th = 20;
    for i = 1 : length(objs)
        
        [~, visible_pt_label] = find_visible_plane(cuboid, objs{i}.intrinsic_params, objs{i}.extrinsic_params * inv(objs{i}.affine_matrx));
        if visible_pt_label(3) || visible_pt_label(4)
            selected_linear_ind{i} = zeros(0);
            continue
        end
        
        z_mat = inv(objs{i}.intrinsic_params * (objs{i}.extrinsic_params * inv(objs{i}.affine_matrx)));
        selector_pos = true(size(visible_pts,1),1); 
        depth_map_ind = objs{i}.frame - depth_cluster.frame_ind(1) + 1;
        
        % Rule out points out of image boundart
        pts_2d_pos = round(project_point_2d(objs{i}.extrinsic_params * inv(objs{i}.affine_matrx), objs{i}.intrinsic_params, [visible_pts(:,1:3), ones(size(visible_pts,1),1)]));
        selector_in_img = pts_2d_pos(:,1) >= 1 & pts_2d_pos(:,1) <= sz_depth_map(2) & pts_2d_pos(:,2) >=1 & pts_2d_pos(:,2) <= sz_depth_map(1);
        pts_2d_pos = pts_2d_pos(selector_in_img, 1:2); visible_pts_ = visible_pts(selector_in_img, :);
        selector_pos(selector_pos) = selector_in_img;
        
        % Rule out positive direction far points
        tot_indices_pos = sub2ind(sz_depth_map, pts_2d_pos(:,2), pts_2d_pos(:,1));
        depth_pos = depth_cluster.depth_maps{depth_map_ind}(tot_indices_pos);
        pts3d_gt_pos = ((z_mat) * [pts_2d_pos(:,1) .* depth_pos, pts_2d_pos(:,2) .* depth_pos, depth_pos, ones(length(depth_pos),1)]')';
       
        % Judge according to differnet plane
        selector_plane1 = (visible_pts_(:,4) == 1); selector_plane1(selector_plane1) = (sqrt(sum((pts3d_gt_pos(selector_plane1, 1:3) - visible_pts(selector_plane1, 1:3)).^2, 2)) < plane_th * w);
        selector_plane2 = (visible_pts_(:,4) == 2); selector_plane2(selector_plane2) = (sqrt(sum((pts3d_gt_pos(selector_plane2, 1:3) - visible_pts(selector_plane2, 1:3)).^2, 2)) < plane_th * l);
        selector_pos(selector_pos) = (selector_plane1 | selector_plane2);
        % visible_pts = visible_pts((selector_plane1 | selector_plane2), :); % For visualization
        % pts3d_gt_pos = pts3d_gt_pos((selector_plane1 | selector_plane2), :); %For visualization
        
        % Exclude border points
        pts_2d_pos = pts_2d_pos((selector_plane1 | selector_plane2), :);
        grad_record = abs([interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pts_2d_pos(:,1) + 1, pts_2d_pos(:,2)]) - interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pts_2d_pos(:,1), pts_2d_pos(:,2)]), interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pts_2d_pos(:,1), pts_2d_pos(:,2) + 1]) - interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pts_2d_pos(:,1), pts_2d_pos(:,2)])]);
        selector_border = (grad_record(:,1) < grad_th) & (grad_record(:,2) < grad_th);
        selector_pos(selector_pos) = selector_border;
        
        
        
        % To sample from depth map
        [yy, xx] = ind2sub(sz_depth_map, objs{i}.linear_ind); pixel_loc = [xx yy];
        depth_inv = depth_cluster.depth_maps{depth_map_ind}(objs{i}.linear_ind);
        x_inv_gt = ((z_mat) * ([xx .* depth_inv, yy .* depth_inv, depth_inv, ones(length(depth_inv), 1)])')';
        % [a, aa] = project_point_2d(objs{i}.extrinsic_params, objs{i}.intrinsic_params, objs{i}.pts_new, objs{i}.affine_matrx);
        
        % Calculate inverse points on cuboid
        [x_inv, plane_type_rec, selector_inv, x11, x22] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, (objs{i}.extrinsic_params * inv(objs{i}.affine_matrx)));
        x_inv_gt = x_inv_gt(selector_inv, :); pixel_loc = pixel_loc(selector_inv, :);
        selector_plane1 = (plane_type_rec == 1); selector_plane1(selector_plane1) = (sqrt(sum((x_inv(selector_plane1, 1:3) - x_inv_gt(selector_plane1, 1:3)).^2, 2)) < plane_th * w);
        selector_plane2 = (plane_type_rec == 2); selector_plane2(selector_plane2) = (sqrt(sum((x_inv(selector_plane2, 1:3) - x_inv_gt(selector_plane2, 1:3)).^2, 2)) < plane_th * l);
        selector_inv(selector_inv) = (selector_plane1 | selector_plane2);
        % x_inv = x_inv((selector_plane1 | selector_plane2), :); % For visualization
        % x_inv_gt = x_inv_gt((selector_plane1 | selector_plane2), :); % For visualization
        
        % Exclude border points
        pixel_loc = pixel_loc((selector_plane1 | selector_plane2), :);
        grad_record = abs([interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pixel_loc(:,1) + 1, pixel_loc(:,2)]) - interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pixel_loc(:,1), pixel_loc(:,2)]), interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pixel_loc(:,1), pixel_loc(:,2) + 1]) - interpImg_(depth_cluster.depth_maps{depth_map_ind}, [pixel_loc(:,1), pixel_loc(:,2)])]);
        selector_border = (grad_record(:,1) < grad_th) & (grad_record(:,2) < grad_th);
        selector_inv(selector_inv) = selector_border;
        [selector_inv, selector_pos] = bi_direction_downsample(selector_inv, selector_pos); tot_valid_pt_num = sum(selector_inv) + sum(selector_pos);
        selected_linear_ind{i} = objs{i}.linear_ind(selector_inv); linaer_ind_selector{i} = selector_inv;
        counts_set(i, :) = selector_pos'; 
        % Calculate valid area
        try
            max_pt_p1(i,:) = [max(x11(selector_plane1, 1)), min(x11(selector_plane1, 1)), max(x11(selector_plane1, 2)), min(x11(selector_plane1, 2))];
            max_pt_p2(i,:) = [max(x22(selector_plane2, 1)), min(x22(selector_plane2, 1)), max(x22(selector_plane2, 2)), min(x22(selector_plane2, 2))];
        catch
            max_pt_p1(i,:) = [-inf, inf, -inf, inf];
            max_pt_p2(i,:) = [-inf, inf, -inf, inf];
        end
        % Visualizaton:
        % cur_depth_map = mark_on_depth_map(depth_cluster.depth_maps{depth_map_ind}, pts_2d_pos(selector_border, 1:2), rand(1,3));
        % figure(1); clf; imshow(cur_depth_map);
        % cur_depth_map = mark_on_depth_map(depth_cluster.depth_maps{depth_map_ind}, pixel_loc(selector_border, 1:2), rand(1,3));
        % figure(2); clf; imshow(cur_depth_map);
        
        % hold on; scatter3(visible_pts(:,1),visible_pts(:,2),visible_pts(:,3),3,'g','fill');
        % hold on; scatter3(pts3d_gt_pos(:,1),pts3d_gt_pos(:,2),pts3d_gt_pos(:,3),3,'b','fill');
        % hold on; scatter3(objs{i}.pts_new(:,1),objs{i}.pts_new(:,2),objs{i}.pts_new(:,3),3,'b','fill');
        %{
        figure(1); clf; draw_cubic_shape_frame(cuboid);
        hold on; scatter3(x_inv(selector_plane1,1),x_inv(selector_plane1,2),x_inv(selector_plane1,3),3,'g','fill');
        hold on; scatter3(x_inv(selector_plane2,1),x_inv(selector_plane2,2),x_inv(selector_plane2,3),3,'b','fill');
        hold on; scatter3(x_inv_gt(:,1),x_inv_gt(:,2),x_inv_gt(:,3),3,'r','fill');
        hold on; scatter3(x_inv_gt(selector_plane1 | selector_plane2,1),x_inv_gt(selector_plane1 | selector_plane2,2),x_inv_gt(selector_plane1 | selector_plane2,3),5,'c','fill');
        axis equal
        %}
        % hold on; scatter3(x_inv_gt(:,1),x_inv_gt(:,2),x_inv_gt(:,3),3,'b','fill');
    end
    max_pt_p1 = [max(max_pt_p1(:,1)), min(max_pt_p1(:,2)), max(max_pt_p1(:,3)), min(max_pt_p1(:,4))];
    max_pt_p2 = [max(max_pt_p2(:,1)), min(max_pt_p2(:,2)), max(max_pt_p2(:,3)), min(max_pt_p2(:,4))];
    area_p1 = (max_pt_p1(1) - max_pt_p1(2)) * (max_pt_p1(3) - max_pt_p1(4)) / l / h;
    area_p2 = (max_pt_p2(1) - max_pt_p2(2)) * (max_pt_p2(3) - max_pt_p2(4)) / w / h;
    if isnan(area_p1) || isnan(area_p2) || tot_valid_pt_num < tot_valid_pt_num_th
        isvalid = false;
    end
    if area_p1 < area_th || area_p2 < area_th
        isvalid = false;
    end
end
function [selector_inv, selector_pos] = bi_direction_downsample(selector_inv, selector_pos)
    if sum(selector_inv) > sum(selector_pos)
        new_num = sum(selector_pos);
        indices = unique(round(linspace(1, sum(selector_inv), new_num)));
        sub_selector = false(sum(selector_inv),1);
        sub_selector(indices) = true;
        selector_inv(selector_inv) = sub_selector;
    else
        new_num = sum(selector_inv);
        indices = unique(round(linspace(1, sum(selector_pos), new_num)));
        sub_selector = false(sum(selector_pos),1);
        sub_selector(indices) = true;
        selector_pos(selector_pos) = sub_selector;
    end
end
function [x, plane_type_rec, selector, x11, x22, d1, d2] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, extrinsic)
    plane_type_rec = zeros(size(pixel_loc,1),1);
    d1 = cal_depth_d(plane_param1', pixel_loc, z_mat); x1 = cal_3d_point_x(pixel_loc, d1, z_mat);
    d2 = cal_depth_d(plane_param2', pixel_loc, z_mat); x2 = cal_3d_point_x(pixel_loc, d2, z_mat);
    x11 = (tans_sign_mat1 * x1')'; x22 = (tans_sign_mat2 * x2')';
    
    overlapped = (x11(:,1) <= l) & (x22(:,1) < w);
    plane_type_rec((x11(:,1) <= l) & (~overlapped)) = 1; plane_type_rec((x22(:,1) < w) & (~overlapped)) = 2;
    if sum(overlapped) > 0
        cam_origin = (inv(extrinsic) * [0 0 0 1]')';
        dist1 = sum((x1(overlapped, 1:3) - repmat(cam_origin(1:3), [sum(overlapped), 1])).^2, 2);
        dist2 = sum((x2(overlapped, 1:3) - repmat(cam_origin(1:3), sum(overlapped), 1)).^2, 2);
        added_on_plane1 = overlapped; added_on_plane1(added_on_plane1) = (dist1 <= dist2);
        added_on_plane2 = overlapped; added_on_plane2(added_on_plane2) = (dist2 < dist1);
        plane_type_rec(added_on_plane1) = 1;
        plane_type_rec(added_on_plane2) = 2;
    end
    
    
    selector = (plane_type_rec ~= 0); selector1 = (plane_type_rec == 1); selector2 = (plane_type_rec == 2);
    x11 = x11(selector,:); x22 = x22(selector,:); d1 = d1(selector); d2 = d2(selector);
    x = zeros(size(x1)); x(selector1, : ) = x1(selector1, :); x(selector2, :) = x2(selector2, :); x = x(selector, :);
    plane_type_rec = plane_type_rec(selector);
end
function x = cal_3d_point_x(pixel_loc, d, z_mat)
    p1 = pixel_loc(:,1); p2 = pixel_loc(:,2);
    x = [
        z_mat(1,1) * p1 .* d + z_mat(1,2) * p2 .* d + z_mat(1,3) .* d + z_mat(1,4), ...
        z_mat(2,1) * p1 .* d + z_mat(2,2) * p2 .* d + z_mat(2,3) .* d + z_mat(2,4), ...
        z_mat(3,1) * p1 .* d + z_mat(3,2) * p2 .* d + z_mat(3,3) .* d + z_mat(3,4), ...
        z_mat(4,1) * p1 .* d + z_mat(4,2) * p2 .* d + z_mat(4,3) .* d + z_mat(4,4), ...
        ];
end
function [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h)
    for plane_ind = 1 : 2
        if plane_ind == 1
            pts_org = [
                xc - 1/2 * l * cos(theta) + 1/2 * w * sin(theta);
                yc - 1/2 * l * sin(theta) - 1/2 * w * cos(theta);
                1/2 * h;
                1
                ];
            pts_x = pts_org + [cos(theta) sin(theta) 0 0]';
            pts_y = pts_org + [0 0 1 0]';
            pts_z = pts_org + [sin(theta) -cos(theta) 0 0]';
            old_pts = [pts_x';pts_y';pts_z';pts_org']; new_pts = [1 0 0 1; 0 1 0 1; 0 0 1 1;0 0 0 1;];
            tans_sign_mat1 = new_pts' * inv(old_pts');
        end
        if plane_ind == 2
            pts_org = [
                xc + 1/2 * l * cos(theta) - 1/2 * w * sin(theta);
                yc + 1/2 * l * sin(theta) + 1/2 * w * cos(theta);
                1/2 * h;
                1
                ];
            pts_x = pts_org + [sin(theta) -cos(theta) 0 0]';
            pts_y = pts_org + [0 0 -1 0]';
            pts_z = pts_org + [cos(theta) sin(theta) 0 0]';
            old_pts = [pts_x';pts_y';pts_z';pts_org']; new_pts = [1 0 0 1; 0 1 0 1; 0 0 1 1;0 0 0 1;];
            tans_sign_mat2 = new_pts' * inv(old_pts');
        end
    end
end
function d = cal_depth_d(plane_param, pixel_loc, z_mat)
    d = - plane_param' * z_mat(:,4) ./ (pixel_loc(:,1) * (plane_param' * z_mat(:,1)) + pixel_loc(:,2) * (plane_param' * z_mat(:,2)) + plane_param' * z_mat(:,3));
end
function cubic_cluster = restore_changed_cubics(cubic_cluster, data_cluster)
    default_activation_label = [1 1 1 1 1 0];
    for i = 1 : length(data_cluster)
        if ~isequal(data_cluster{i}{1}.restore_matrix, eye(4))
            [cuboid, visible_pts] = restore_new_cubic_shape(cubic_cluster{i}.cuboid, data_cluster{i}{1});
            cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = visible_pts;
            cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
            cubic_cluster{i}.activation_label = default_activation_label;
        end
    end
    for i = 1 : length(data_cluster)
        [cuboid, visible_pts] = re_estimate_cubic_entry(cubic_cluster{i}.cuboid, data_cluster{i});
        cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = visible_pts;
        cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
        cubic_cluster{i}.activation_label = default_activation_label;
    end
end
function [cuboid, sampled_pts] = re_estimate_cubic_entry(old_cuboid, objs)
    % Adjust initial guess for the new coming points
    tot_pt_num = 0; iou_th = 0.7;
    for i = 1 : length(objs)
        tot_pt_num = tot_pt_num + length(objs{i}.linear_ind);
    end
    tot_pts3d = zeros(tot_pt_num, 4); cur_pos = 1;
    for i = 1 : length(objs)
        tot_pts3d(cur_pos : cur_pos + length(objs{i}.linear_ind) - 1, :) = objs{i}.pts_new;
        cur_pos = cur_pos + length(objs{i}.linear_ind);
    end
    [~, new_cuboid] = estimate_rectangular(tot_pts3d);
    iou = calculate_IOU(old_cuboid, new_cuboid);
    if iou < iou_th
        cuboid_fin = new_cuboid;
    else
        cuboid_fin = old_cuboid;
    end
    [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid_fin, objs{end});
end
function [cuboid, sampled_pts] = restore_new_cubic_shape(old_cubic, cur_obj)
    restore_matrix = cur_obj.restore_matrix; affine_matrix = cur_obj.affine_matrx;
    params = generate_cubic_params(old_cubic);
    bottom_pts = old_cubic{5}.pts(1:4,:); bottom_pts(:,3) = 0; bottom_pts = [bottom_pts ones(size(bottom_pts, 1),1)];
    bottom_pts = (affine_matrix * restore_matrix * bottom_pts')'; mean_pts = mean(bottom_pts); cx = mean_pts(1); cy = mean_pts(2); 
    params(2) = cx; params(3) = cy; 
    pts_on_first_plane = old_cubic{1}.pts(1:2, :); pts_on_first_plane = [pts_on_first_plane ones(size(pts_on_first_plane,1),1)]; 
    pts_on_first_plane = (affine_matrix * restore_matrix * pts_on_first_plane')';
    pts_on_first_plane = pts_on_first_plane(:,1:2); dir_vector = pts_on_first_plane(2,:) - pts_on_first_plane(1,:);
    dir_vector = dir_vector / norm(dir_vector); S = dir_vector(2); C = dir_vector(1);
    theta = angleCalc(S, C, 'rad');
    params(1) = theta;
    new_cuboid = generate_center_cuboid_by_params(params);
    [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(new_cuboid, cur_obj);
end
function [data_cluster, cubic_cluster] = trim_incontinuous_frame(data_cluster, cubic_cluster)
    for i = 1 : length(data_cluster)
        cuboid = cubic_cluster{i}.cuboid; [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid, data_cluster{i}{end});
        cubic_cluster{i}.cuboid = cuboid; cubic_cluster{i}.visible_pts = sampled_pts;
        is_valid_array = fill_is_valid(cuboid, data_cluster{i}); cubic_cluster{i}.is_valid = is_valid_array;
    end
end
function is_valid_array = fill_is_valid(cuboid, entries)
    is_valid_array = zeros(length(entries), 1);
    for i = 1 : length(entries)
        is_valid_array(i) = judge_is_still_visible(cuboid, entries{i});
    end
end
function [cuboid, sampled_pts] = tune_cubic_shape_to_specific_entry(cuboid, cur_obj)
    extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic);
    sampled_pts = acquire_visible_sampled_points(cuboid, cur_obj.intrinsic_params, cur_obj.extrinsic_params, cur_obj.affine_matrx);
end
function is_visible = judge_is_still_visible(cubic, cur_obj)
    extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    is_visible = jude_is_first_and_second_plane_visible(cubic, intrinsic, extrinsic);
end
function new_cubic_cluster = get_cubic_cluster(data_cluster, cubic_cluster)
    new_cubic_cluster = cell(length(data_cluster), 1); default_activation_label = [1 1 1 1 1 0];
    for i = 1 : length(data_cluster)
        pos = find_pos_of_certain_ind_in_cubic_cluster(cubic_cluster, data_cluster{i}{1}.instanceId);
        if pos == 0
            [cuboid, visible_pts, is_one_plane_visible] = estimate_cubic_shape_by_obj(data_cluster{i}{1});
            new_cubic_cluster{i}.cuboid = cuboid; new_cubic_cluster{i}.visible_pts = visible_pts;
            new_cubic_cluster{i}.instanceId = data_cluster{i}{1}.instanceId;
            new_cubic_cluster{i}.activation_label = default_activation_label;
            new_cubic_cluster{i}.is_valid = is_one_plane_visible;
        else
            new_cubic_cluster{i} = cubic_cluster{pos};
        end
    end
end
function [cuboid, sampled_pts, is_valid] = estimate_cubic_shape_by_obj(cur_obj)
    pts_3d = cur_obj.pts_new; extrinsic = cur_obj.extrinsic_params * inv(cur_obj.affine_matrx); intrinsic = cur_obj.intrinsic_params;
    [~, cuboid] = estimate_rectangular(pts_3d);
    [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic);
    sampled_pts = acquire_visible_sampled_points(cuboid, cur_obj.intrinsic_params, cur_obj.extrinsic_params, cur_obj.affine_matrx);
    % figure(1); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(pts_3d(:,1), pts_3d(:,2), pts_3d(:,3), 3, 'g', 'fill')
    % hold on; scatter3(sampled_pts(:,1),sampled_pts(:,2),sampled_pts(:,3),3,'b','fill')
end
function pos = find_pos_of_certain_ind_in_cubic_cluster(cubic_cluster, instance_id)
    pos = 0;
    for i = 1 : length(cubic_cluster)
        if cubic_cluster{i}.instanceId == instance_id
           pos = i;
        end
    end
end

function [data_cluster, depth_cluster] = read_in_clusters(frame, help_info_entry)
    path = [help_info_entry{9} 'Instance_map_organized/'];
    holder1 = load([path num2str(frame, '%06d') '.mat']); data_cluster = holder1.data_cluster; data_cluster = correct_affine_error(data_cluster);
    holder2 = load([path num2str(frame, '%06d') '_d.mat']); depth_cluster = holder2.depth_cluster;
end
function data_cluster = correct_affine_error(data_cluster)
    for i = 1 : length(data_cluster)
        for j = 2 : length(data_cluster{i})
            data_cluster{i}{j}.affine_matrx = data_cluster{i}{1}.affine_matrx;
        end
    end
end
function print_matrix(fileID, m)
    for i = 1 : size(m,1)
        for j = 1 : size(m,2)
            fprintf(fileID, '%d\t', m(i,j));
        end
        fprintf(fileID, '\n');
    end
end
function path = make_dir(help_info_entry)
    global path_mul
    father_folder = [help_info_entry{8}];
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path_mul = [father_folder DateString '_mul/'];
    mkdir(path_mul); path = path_mul; mkdir([path_mul 'rgb_image/']);
end
function params = generate_cubic_params(cuboid)
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h];
end
function sampled_pts = acquire_visible_sampled_points(cuboid, intrinsic_params, extrinsic_params, affine_matrix)
    sample_pt_num = 15;
    sampled_pts = sample_cubic_by_num(cuboid, sample_pt_num, sample_pt_num);
    extrinsic_params = extrinsic_params * inv(affine_matrix);
    visible_label = find_visible_pt_global_({cuboid}, [sampled_pts(:, 1:3) ones(size(sampled_pts,1),1)], intrinsic_params, extrinsic_params);
    % visible_label = find_visible_pt_global({cuboid}, sampled_pts(:, 1:3), intrinsic_params, extrinsic_params);
    sampled_pts = sampled_pts(visible_label, :);
end
function img = cubic_lines_of_2d(img, cubic, intrinsic_params, extrinsic_params)
    % color = uint8(randi([1 255], [1 3]));
    % color = rand([1 3]);
    shapeInserter = vision.ShapeInserter('Shape', 'Lines', 'BorderColor', 'Custom');
    pts3d = zeros(8,4); mh = cubic{1}.mh;
    for i = 1 : 4
        pts3d(i, :) = [cubic{i}.pts(1, :) 1];
        pts3d(i, 3) = pts3d(i, 3) + mh;
    end
    for i = 5 : 8
        pts3d(i, :) = [cubic{5}.pts(i - 4, :) 1];
    end
    pts2d = (intrinsic_params * extrinsic_params * [pts3d(:, 1:3) ones(size(pts3d,1),1)]')';
    depth = pts2d(:,3);
    pts2d(:, 1) = pts2d(:,1) ./ depth; pts2d(:,2) = pts2d(:,2) ./ depth; pts2d = round(pts2d(:,1:2));
    lines = zeros(12, 4);
    lines(4, :) = [pts2d(4, :) pts2d(1, :)];
    lines(12, :) = [pts2d(5, :) pts2d(8, :)];
    for i = 1 : 3
        lines(i, :) = [pts2d(i, :) pts2d(i+1, :)];
    end
    for i = 1 : 4
        lines(4 + i, :) = [pts2d(i, :), pts2d(i + 4, :)];
    end
    for i = 1 : 3
        lines(8 + i, :) = [pts2d(i + 4, :) pts2d(i + 5, :)];
    end
    for i = 1 : 12
        img = step(shapeInserter, img, int32([lines(i, 1) lines(i, 2) lines(i, 3) lines(i, 4)]));
    end
end
function cuboid = generate_center_cuboid_by_params(params)
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
end
function alternate_cuboids = transfer_cuboid(cuboid)
    alternate_cuboids = cell(4,1);
    params = generate_cubic_params(cuboid);
    for i = 1 : 4
        if i == 1
            cur_params = params;
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 2
            cur_params = params; cur_params(1) = cur_params(1) + pi / 2;
            cur_params(4) = params(5); cur_params(5) = params(4);
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 3
            cur_params = params; cur_params(1) = cur_params(1) + pi;
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
        if i == 4
            cur_params = params; cur_params(1) = cur_params(1) + pi / 2 * 3;
            cur_params(4) = params(5); cur_params(5) = params(4);
            alternate_cuboids{i} = generate_center_cuboid_by_params(cur_params);
        end
    end
end
function [cuboid, is_valid] = tune_cuboid(cuboid, extrinsic, intrinsic)
    alternate_cuboids = transfer_cuboid(cuboid);
    judge_re = false(length(alternate_cuboids),1); is_valid = false;
    for i = 1 : length(judge_re)
        judge_re(i) = jude_is_first_and_second_plane_visible(alternate_cuboids{i}, intrinsic, extrinsic);
    end
    ind = find(judge_re);
    if length(ind) > 0
        cuboid = alternate_cuboids{ind};
        is_valid = true;
    end
end
function is_visible = jude_is_first_and_second_plane_visible(cuboid, intrinsic_params, extrinsic_params)
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    c1 = [
        xc + 1 / 2 * w * sin(theta);
        yc - 1 / 2 * w * cos(theta);
        h/2;
        1;
        ];
    c2 = [
        xc + 1/2 * l * cos(theta);
        yc + 1/2 * l * sin(theta);
        1/2 * h;
        1
        ];
    visible_pt_label = find_visible_pt_global_({cuboid}, [c1';c2'], intrinsic_params, extrinsic_params);
    % visible_pt_label = find_visible_pt_global({cuboid}, [c1';c2'], intrinsic_params, extrinsic_params);
    is_visible = visible_pt_label(1) & visible_pt_label(2); 
end