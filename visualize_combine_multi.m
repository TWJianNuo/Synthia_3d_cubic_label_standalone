function [x_inv_record, dominate_pts, dominate_color] = visualize_combine_multi(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pt_3d, activation_label, color1, color2)
    % load('debug.mat')
    global sigmoid_m sigmoid_bias
    sigmoid_m = 10; sigmoid_bias = 4;
    ratio = 1.5; activation_label = (activation_label == 1); [flag1, flag2] = judge_flag(nargin); flag1 = false; flag2 = false;
    if ~flag1 color1 = [zeros(length(linear_ind),1) zeros(length(linear_ind),1) ones(length(linear_ind),1)]; end
    if ~flag2 color2 = [zeros(size(visible_pt_3d, 1),1) ones(size(visible_pt_3d, 1),1) zeros(size(visible_pt_3d, 1),1)]; end
    [params, gt, pixel_loc] = make_preparation(cuboid, extrinsic_param, intrinsic_param, linear_ind, depth_map);
    pts_3d_gt = calculate_ed_pts(extrinsic_param, intrinsic_param, linear_ind, depth_map(linear_ind), size(depth_map));
    
    [plane_ind_batch, cuboid, selector] = sub_preparation(intrinsic_param, extrinsic_param, pixel_loc, params);
    pixel_loc = pixel_loc(selector, :); linear_ind = linear_ind(selector); plane_ind_batch = plane_ind_batch(selector); pts_3d_gt = pts_3d_gt(selector,:); color1 = color1(selector,:);
    
    grad_inv_record = get_grad_inv(cuboid, intrinsic_param, extrinsic_param, pixel_loc, activation_label, gt, plane_ind_batch);
    x_inv_record = get_x_inv(cuboid, intrinsic_param, extrinsic_param, pixel_loc, activation_label, gt, plane_ind_batch);
    [grad_pos_record, x_pos_record] = get_grad_pos(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map, ratio);
    
    diff_inv_record = get_diff_inv(cuboid, pixel_loc, intrinsic_param, extrinsic_param, gt, plane_ind_batch);
    [diff_pos_record, pts_pos_gt] = get_diff_pos(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params, ratio);
    [dominate_pts, dominate_color] = visualize(cuboid, x_inv_record, x_pos_record, grad_inv_record, grad_pos_record, diff_inv_record, diff_pos_record, visible_pt_3d, intrinsic_param, extrinsic_param, plane_ind_batch, pixel_loc, activation_label, pts_3d_gt, pts_pos_gt, color1, color2);
end
function [flag1, flag2] = judge_flag(num_input)
    flag1 = false; flag2 = false;
    if num_input == 8
        flag1 = true;
    end
    if num_input == 9
        flag2 = true; flag1 = true;
    end
end
function pts_3d = calculate_ed_pts(extrinsic_params, intrinsic_params, linear_ind, depth, sz_depth)
    [yy, xx] = ind2sub([sz_depth(1) sz_depth(2)], linear_ind); pixel_2d = [xx yy]; % lin_check = sub2ind(sz_depth, pixel_2d(:,2), pixel_2d(:,1));
    tmp = [pixel_2d(:,1) .* depth, pixel_2d(:,2) .* depth, depth, ones(size(depth,1),1)];
    pts_3d = (inv(intrinsic_params * extrinsic_params) * tmp')';
end
function [dominate_pts, dominate_color] = visualize(cuboid, x_inv_record, x_pos_record, grad_inv_record, grad_pos_record, diff_inv_record, diff_pos_record, visible_pt_3d, intrinsic_param, extrinsic_param, plane_ind_batch, pixel_loc_batch, activation_label, pts_3d_gt, pts_pos_gt, color1, color2)
    max_ratio = 10; params_org = generate_cubic_params(cuboid); params_org_ = params_org(activation_label);
    grad_inv_record = grad_inv_record / norm(grad_inv_record) / max_ratio;
    grad_pos_record = grad_pos_record / norm(grad_pos_record) / max_ratio;
    
    params1 = repmat(params_org, [size(grad_inv_record, 1) 1]); params2 = repmat(params_org, [size(grad_pos_record, 1) 1]);
    params1(:, activation_label) = params1(:, activation_label) + grad_inv_record; 
    params2(:, activation_label) = params2(:, activation_label) + grad_pos_record;
    
    pts3_inv_change = zeros(size(grad_inv_record, 1), 4);
    pts3_pos_change = zeros(size(grad_pos_record, 1), 4);
    
    for i = 1 : size(pts3_inv_change, 1)
        pts3_inv_change(i,:) = generate_3d_pts_inv(intrinsic_param, extrinsic_param, pixel_loc_batch(i,:), plane_ind_batch(i), generate_center_cuboid_by_params(params1(i,:)));
    end
    
    for i = 1 : size(pts3_pos_change, 1)
        k = [visible_pt_3d(i,1) visible_pt_3d(i,2)]; plane_ind = visible_pt_3d(i,3);
        pts3_pos_change(i,:) = [generate_3d_pts_pos(params2(i,:), k, plane_ind) 1];
    end
    
    dir_inv = pts3_inv_change(:,1:3) - x_inv_record(:,1:3); dir_inv = dir_inv ./ repmat(vecnorm(dir_inv, 2, 2), [1 3]);
    dir_pos = pts3_pos_change(:,1:3) - x_pos_record(:,1:3); dir_pos = dir_pos ./ repmat(vecnorm(dir_pos, 2, 2), [1 3]);
    
    val = [diff_inv_record; diff_pos_record]; colors = generate_cmap_array(val); pts = [x_inv_record; x_pos_record];
    quiv_size = (val - min(val)); quiv_size = quiv_size / max(quiv_size) * 3 + 0.5; 
    quiv_size_inv = quiv_size(1: size(dir_inv,1)); quiv_size_pos = quiv_size(size(dir_inv,1) + 1 : end);
    
    dominate_pts = [x_inv_record; x_pos_record]; dominate_selector = (quiv_size > 1); dominate_pts = dominate_pts(dominate_selector, :);
    dominate_color = [color1; color2]; dominate_color = dominate_color(dominate_selector, :);
    figure(2); clf;
    draw_cubic_shape_frame(cuboid); hold on;
    scatter3(x_inv_record(:,1),x_inv_record(:,2),x_inv_record(:,3),20,color1,'fill'); hold on;
    % pts_c = [pts_pos_gt; pts_3d_gt];
    pts_c = [pts_3d_gt];
    scatter3(pts_c(:,1),pts_c(:,2),pts_c(:,3),20,'c','fill'); hold on;
    % scatter3(x_pos_record(:,1),x_pos_record(:,2),x_pos_record(:,3),20,color2,'fill'); hold on;
    for i = 1 : size(x_inv_record,1)
        quiver3(x_inv_record(i,1),x_inv_record(i,2),x_inv_record(i,3),dir_inv(i,1),dir_inv(i,2),dir_inv(i,3),quiv_size_inv(i),'b'); hold on;
    end
    for i = 1 : size(x_pos_record,1)
        % quiver3(x_pos_record(i,1),x_pos_record(i,2),x_pos_record(i,3),dir_pos(i,1),dir_pos(i,2),dir_pos(i,3),quiv_size_pos(i),'g'); hold on;
    end
    for i = 1 : size(x_inv_record,1)
        plot3([x_inv_record(i,1);pts_3d_gt(i,1)],[x_inv_record(i,2);pts_3d_gt(i,2)],[x_inv_record(i,3);pts_3d_gt(i,3)],'LineStyle',':','Color','k'); hold on;
    end
    for i = 1 : size(x_pos_record,1)
        % plot3([x_pos_record(i,1);pts_pos_gt(i,1)],[x_pos_record(i,2);pts_pos_gt(i,2)],[x_pos_record(i,3);pts_pos_gt(i,3)],'LineStyle',':','Color','k'); hold on;
    end
end
function colors = generate_cmap_array(val)
    cmap = colormap(); val = val - min(val) + 0.1;
    colors = cmap(ceil(val / max(val) * (size(cmap,1) - 1)), :);
end
function pts3 = generate_3d_pts_inv(intrinsic_param, extrinsic_param, pixel_loc, plane_ind, cuboid)
    cur_plane_param = get_plane_param(cuboid, plane_ind);
    d = cal_depth_d(cur_plane_param, intrinsic_param, extrinsic_param, pixel_loc);
    pts3 = cal_3d_point_x(pixel_loc, d, intrinsic_param, extrinsic_param)';
end
function pts3 = generate_3d_pts_pos(params, k, plane_ind)
    pts3 = pts_3d_(params, k, plane_ind)';
end
function [loss, diff, grad] = regularized_term_grad_loss(ratio, activation_label)
    loss = 0; diff = 0;
    grad = eye(sum(activation_label)) * ratio;
end
%{
function [best_cuboid, max_iou, best_params] = find_best_cuboid(params_record, cuboid_gt)
    iou_record = zeros(length(params_record), 1);
    for i = 1 : length(params_record)
        cuboid_cur = generate_center_cuboid_by_params(params_record(i,:));
        iou_record(i) = calculate_IOU(cuboid_gt, cuboid_cur);
    end
    ind_max = find(iou_record == max(iou_record)); ind_max = ind_max(1);
    best_cuboid = generate_center_cuboid_by_params(params_record(ind_max,:));
    max_iou = max(iou_record); best_params = params_record(ind_max,:);
end
%}
function [best_cuboid, max_iou, best_params] = find_best_cuboid(params_record, cuboid_gt, diff_record)
    best_ind = find(diff_record == min(diff_record)); best_ind = best_ind(1);
    best_cuboid = generate_center_cuboid_by_params(params_record(best_ind,:));
    max_iou = calculate_IOU(cuboid_gt, best_cuboid); best_params = params_record(best_ind,:);
end
function [best_params, depth_map, space_map, stem_map, max_iou] = organize_reults(params_record, cuboid_gt, intrinsic_param, extrinsic_param, depth_map, visible_pt_3d, diff_record)
    if isempty(diff_record)
        best_params = zeros(0);
        depth_map = zeros(0);
        space_map = zeros(0);
        stem_map = zeros(0); 
        max_iou = zeros(0);
    else
        [best_cuboid, max_iou, best_params] = find_best_cuboid(params_record, cuboid_gt, diff_record);
        depth_map = visualize_re(cuboid_gt, best_cuboid, intrinsic_param, extrinsic_param, depth_map);
        space_map = plot_scene(cuboid_gt, best_params, visible_pt_3d);
        stem_map = plot_stem(diff_record);
    end
end
function X = plot_scene(old_cuboid, params, visible_pt_3d)
    % figure(1); clf; scatter3(pts_new(:, 1), pts_new(:, 2), pts_new(:, 3), 3, 'g', 'fill'); hold on;
    f = figure('visible', 'off'); clf;
    cuboid = generate_cuboid_by_center(params(2), params(3), params(1), params(4), params(5), params(6)); draw_cubic_shape_frame(cuboid); sampled_pts = sample_cubic_by_num(cuboid, 10, 10);
    selector = (sampled_pts(:, 4) == visible_pt_3d(1, 3) | sampled_pts(:, 4) == visible_pt_3d(end, 3));
    sampled_pts = sampled_pts(selector, :);
    hold on; scatter3(sampled_pts(:,1), sampled_pts(:,2), sampled_pts(:,3), 3, 'r', 'fill')
    hold on; draw_cuboid(old_cuboid); F = getframe(f); [X, Map] = frame2im(F);
end
function X = plot_stem(diff_record)
    f = figure('visible', 'off'); stem(diff_record(diff_record~=0), 'fill'); F = getframe(f); [X, Map] = frame2im(F);
end
function [plane_ind_batch, cuboid, selector] = sub_preparation(intrinsic_param, extrinsic_param, pixel_loc, params)
    cuboid = generate_center_cuboid_by_params(params);
    cuboid = add_transformation_matrix_to_cuboid(cuboid);
    [plane_ind_batch, selector] = judege_plane(cuboid, intrinsic_param, extrinsic_param, pixel_loc);
    
end
function [params, gt, pixel_loc] = make_preparation(cuboid, extrinsic_param, intrinsic_param, linear_ind, depth_map)
    gt = get_ground_truth(depth_map, linear_ind); pixel_loc = get_pixel_loc(depth_map, linear_ind);
    params = generate_cubic_params(cuboid);
end
function x_record = get_x_inv(cuboid, intrinsic_param, extrinsic_param, pixel_loc_batch, activation_label, ground_truth, plane_ind_batch)
    tot_num = size(pixel_loc_batch, 1); x_record = zeros(size(pixel_loc_batch, 1), 4);
    for i = 1 : tot_num
        pixel_loc = pixel_loc_batch(i,:); plane_ind = plane_ind_batch(i);
        cur_plane_param = get_plane_param(cuboid, plane_ind);
        d = cal_depth_d(cur_plane_param, intrinsic_param, extrinsic_param, pixel_loc);
        x_record(i, :) = cal_3d_point_x(pixel_loc, d, intrinsic_param, extrinsic_param)';
    end
end
function grad_record = get_grad_inv(cuboid, intrinsic_param, extrinsic_param, pixel_loc_batch, activation_label, ground_truth, plane_ind_batch)
    params = generate_cubic_params(cuboid);
    tot_num = size(pixel_loc_batch, 1); sum_diff = zeros(1); sum_hess = zeros(sum(activation_label)); sum_jacob = zeros(sum(activation_label));
    grad_record =  zeros(size(pixel_loc_batch, 1), sum(activation_label));
    for i = 1 : tot_num
        pixel_loc = pixel_loc_batch(i,:); plane_ind = plane_ind_batch(i);
        grad_gt_theta = get_grad_gt_theta(cuboid, pixel_loc, intrinsic_param, extrinsic_param, plane_ind, params);
        grad_ft_theta = get_grad_ft_theta(cuboid, pixel_loc, intrinsic_param, extrinsic_param, plane_ind, params);
        gt = get_gt(cuboid, pixel_loc, intrinsic_param, extrinsic_param, plane_ind);
        ft = get_ft(cuboid, pixel_loc, intrinsic_param, extrinsic_param, plane_ind, params);
        jacob = grad_gt_theta * ft + grad_ft_theta * gt; jacob = jacob(activation_label);
        grad_record(i, :) = jacob;
    end
end
function delta = get_delta_from_diff_and_hess(sum_diff, sum_hess, activation_label)
    if isnan(sum_diff) | isnan(sum_hess)
        delta = 0;
    else
        delta = smooth_hessian(sum_diff, sum_hess, activation_label);
    end
end

function depth_map = visualize_re(init_cubic, resulted_cubic, intrinsic_params, extrinsic_params, depth_map)
    depth_map = depth_map / max(max(depth_map));
    % figure(2); clf; draw_cuboid(init_cubic); hold on; draw_cubic_shape_frame(resulted_cubic); hold on;
    % sampled_pts_init = acquire_visible_sampled_points(init_cubic, intrinsic_params, extrinsic_params);
    sampled_pts_re = acquire_visible_sampled_points(resulted_cubic, intrinsic_params, extrinsic_params);
    % pts_2d_init = transfer_3d_pts_to_2d(intrinsic_params, extrinsic_params, sampled_pts_init(:,1:3));
    pts_2d_re = transfer_3d_pts_to_2d(intrinsic_params, extrinsic_params, sampled_pts_re(:,1:3));
    % depth_map = mark_depth_map(depth_map, pts_2d_init, 1);
    depth_map = mark_depth_map(depth_map, pts_2d_re, 2);
end
function depth_map = mark_depth_map(depth_map, pts2d, flag)
    if size(depth_map,3) == 1
        depth_map = cat(3, depth_map, depth_map, depth_map);
    end
    if flag == 1
        depth_map = insertMarker(depth_map, pts2d, '*', 'color', 'red', 'size', 1);
    else
        depth_map = insertMarker(depth_map, pts2d, '*', 'color', 'green', 'size', 1);
    end
end
function pts_2d = transfer_3d_pts_to_2d(intrinsic_params, extrinsic_params, pts_3d)
    [pts_2d, ~] = project_point_2d(extrinsic_params, intrinsic_params, pts_3d);
    pts_2d = ceil(pts_2d);
end
function sampled_pts = acquire_visible_sampled_points(cuboid, intrinsic_params, extrinsic_params)
    sample_pt_num = 10;
    sampled_pts = sample_cubic_by_num(cuboid, sample_pt_num, sample_pt_num);
    visible_label = find_visible_pt_global({cuboid}, sampled_pts(:, 1:3), intrinsic_params, extrinsic_params);
    sampled_pts = sampled_pts(visible_label, :);
end
function to_stop = judge_stop(delta, params, diff, stop_flag)
    th = 0.0001; to_stop = false; diff = diff(diff~=0); step_range = 10; th_hold = 0.1;
    if stop_flag
        to_stop = true;
    end
    if max(abs(delta)) < th | params(4) < 0 | params(5) < 0
        to_stop = true;
    end
    if length(diff) > step_range
        if abs(diff(end) - diff(end - step_range)) < th_hold
            to_stop = true;
        end
    end
end
function judge_tune_and_plane_allocation_func(cuboid, intrinsic_param, extrinsic_param, pixel_loc, depth_map, linear_ind)
    exp_num = 200; path = make_dir();
    for i = 1 : exp_num
        cur_cuboid = rotate_degree_to_cuboid(cuboid, rand(1) * 2 * pi);
        [cur_cuboid, is_valid] = tune_cuboid(cur_cuboid, extrinsic_param, intrinsic_param);
        if is_valid
            plane_ind_batch = judege_plane(cur_cuboid, intrinsic_param, extrinsic_param, pixel_loc);
            if length(find(plane_ind_batch == 0)) == 0
                f = figure(1); clf;
                visualize_3d(extrinsic_param, intrinsic_param, cur_cuboid, depth_map, linear_ind, plane_ind_batch); view([-70,25])
                F = getframe(f); [X, Map] = frame2im(F); imwrite(X, [path '/' num2str(i) '.png']);
            else
                disp('Error');
            end
        end
    end
end
function test(cuboid)
    params = generate_cubic_params(cuboid);
    for deg = 0 : pi : pi
        cur_params = params; cur_params(1) = params(1) + deg;
        cuboid = generate_center_cuboid_by_params(cur_params);
        figure(1); hold on; draw_cubic_shape_frame(cuboid);
    end
end

function plane_val = judge_valid_points_on_cubic(pixel_loc, cuboid, extrinsic, intrinsic)
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    plane1_val = false(size(pixel_loc,1),1); plane2_val = false(size(pixel_loc,1),1);
    for plane_ind = 1 : 2
        cur_plane_param = get_plane_param(cuboid, plane_ind);
        d = cal_depth_d(cur_plane_param, intrinsic, extrinsic, pixel_loc);
        if plane_ind == 1
            x1 = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic)';
        end
        if plane_ind == 2
            x2 = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic)';
        end
    end
    for plane_ind = 1 : 2
        if plane_ind == 1
            A = get_transformation_matrix(cuboid, plane_ind);
            x1 = (A * x1')'; selector = (x1(:, 1) >= 0 && x1(:, 1) <= l);
            plane1_val(selector) = true;
        end
        if plane_ind == 2
            A = get_transformation_matrix(cuboid, plane_ind);
            x2 = (A * x2')'; selector = (x2(:,1) >= 0 && x2(:, 1) <= w);
            plane2_val(selector) = 2;
        end
    end
    plane_val = plane1_val | plane2_val;
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
    visible_pt_label = find_visible_pt_global({cuboid}, [c1';c2'], intrinsic_params, extrinsic_params);
    is_visible = visible_pt_label(1) & visible_pt_label(2); 
    % figure(1); hold on;
    % rand_color = rand(1,3); c = [c1';c2';];
    % scatter3(c(:,1),c(:,2),c(:,3),40,rand_color,'fill'); hold on; draw_cubic_shape_frame(cuboid)
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
function visualize_alternate_cuboids(cuboid_sets)
    figure(1); clf;
    for i = 1 : length(cuboid_sets)
        figure(1); hold on; draw_cubic_shape_frame(cuboid_sets{i});
    end
end
function cuboid = rotate_degree_to_cuboid(cuboid, degree)
    params = generate_cubic_params(cuboid); 
    params(1) = params(1) + degree;
    cuboid = generate_center_cuboid_by_params(params);
end
function sign_rec = judge_sign(cuboid, pts_3d, plane_ind)
    global sigmoid_m sigmoid_bias
    th = - sigmoid_bias / sigmoid_m;
    A = get_transformation_matrix(cuboid, plane_ind);
    pts_3d = (A * pts_3d')'; sign_rec = zeros(size(pts_3d,1),1);
    sign_rec(pts_3d(:,1)<=th) = -1; sign_rec(pts_3d(:,1)>th) = 1;
end
function [plane_type_rec, selector] = judege_plane(cuboid, intrinsic, extrinsic, pixel_loc)
    % figure(1); clf;
    params = generate_cubic_params(cuboid); stop_flag = false;
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    plane_type_rec = zeros(size(pixel_loc,1),1);
    for plane_ind = 1 : 2
        cur_plane_param = get_plane_param(cuboid, plane_ind);
        d = cal_depth_d(cur_plane_param, intrinsic, extrinsic, pixel_loc);
        if plane_ind == 1
            x1 = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic)';
        end
        if plane_ind == 2
            x2 = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic)';
        end
    end
    for plane_ind = 1 : 2
        if plane_ind == 1
            A = get_transformation_matrix(cuboid, plane_ind);
            % draw_cuboid(cuboid); hold on; cmap = colormap; color_map_ind = map_vector_to_colormap(x1_sv(:,1), cmap);
            % scatter3(x1_sv(:,1), x1_sv(:,2), x1_sv(:,3), 3, cmap(color_map_ind, :), 'fill');
            % x1_sv = x1;
            x1 = (A * x1')'; selector = x1(:,1) <= l; 
            plane_type_rec(selector) = 1;
        end
        if plane_ind == 2
            A = get_transformation_matrix(cuboid, plane_ind);
            % draw_cuboid(cuboid); hold on; cmap = colormap; color_map_ind = map_vector_to_colormap(x2_sv(:,1), cmap);
            % scatter3(x2_sv(:,1), x2_sv(:,2), x2_sv(:,3), 3, cmap(color_map_ind, :), 'fill');
            % x2_sv = x2;
            x2 = (A * x2')'; selector = x2(:,1) < w;
            plane_type_rec(selector) = 2;
        end
    end
    if sum(plane_type_rec~=0) ~= length(plane_type_rec)
        % stop_flag = true;
        selector = (plane_type_rec~=0);
    else
        selector = true(length(plane_type_rec), 1);
    end
end
function cuboid = add_transformation_matrix_to_cuboid(cuboid)
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
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
            A = new_pts' * inv(old_pts'); cuboid{plane_ind}.inverse_dir_trans_matrix = A;
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
            A = new_pts' * inv(old_pts'); cuboid{plane_ind}.inverse_dir_trans_matrix = A;
        end
    end
end
function A = get_transformation_matrix(cuboid, plane_ind)
    A = cuboid{plane_ind}.inverse_dir_trans_matrix;
    %{
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
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
    end
    old_pts = [pts_x';pts_y';pts_z';pts_org']; new_pts = [1 0 0 1; 0 1 0 1; 0 0 1 1;0 0 0 1;];
    A = new_pts' * inv(old_pts');
    %}
end
function grad_loss = sum_grad(cuboid, pixel_loc, intrinsic_param, extrinsic_param, plane_ind_batch, gt, activation_label)
    grad_loss = zeros(1, sum(activation_label));
    for i = 1 : length(gt)
        plane_ind = plane_ind_batch(i);
        tmp_loss = get_grad_loss(cuboid, pixel_loc(i,:), intrinsic_param, extrinsic_param, plane_ind, gt(i));
        grad_loss = grad_loss + tmp_loss(activation_label);
    end
end
function linear_ind = down_sample_data(linear_ind)
    sampled_num = 200; sample_rate = round(length(linear_ind) / sampled_num);
    if sample_rate == 0
        return
    end
    linear_ind = linear_ind(1 : sample_rate : end);
end
function params = update_param(params, delta_theta, activation_label)
    ratio = max(abs(delta_theta) ./ (0.05 * abs(params(activation_label))));
    if ratio > 1
        delta_theta = delta_theta / ratio;
    end
    params(activation_label) = params(activation_label) + delta_theta;
end
function delta_theta = smooth_hessian(sum_diff, sum_hessian, activation_label)
    delta_theta = sum_diff * inv(sum_hessian);
end
function delta = get_delta(cuboid, pixel_loc_batch, ground_truth, intrinsic, extrinsic, activation_label, plane_ind_batch)
    tot_num = size(pixel_loc_batch, 1); sum_diff = zeros(1); sum_hess = zeros(sum(activation_label));
    for i = 1 : tot_num
        pixel_loc = pixel_loc_batch(i,:); plane_ind = plane_ind_batch(i);
        
        %{
        is_right1 = check_grad_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        is_right2 = check_grad_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        is_right3 = check_grad_loss(cuboid, pixel_loc, intrinsic, extrinsic, ground_truth(i), plane_ind);
        if ~is_right1 | ~is_right2 | ~is_right3
            a = 1;
        end
        %}
        
        grad_gt_theta = get_grad_gt_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        grad_ft_theta = get_grad_ft_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        gt = get_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        ft = get_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
        jacob = grad_gt_theta * ft + grad_ft_theta * gt; jacob = jacob(activation_label);
        sum_diff = sum_diff + (ground_truth(i) - gt * ft) * jacob;
        sum_hess = sum_hess + jacob' * jacob;
    end
    % delta = sum_diff * inv(sum_hess + eye(sum(activation_label)) * num_stable);
    % delta = sum_diff * inv(sum_hess);
    if isnan(sum_diff) | isnan(sum_hess)
        delta = 0;
    else
        delta = smooth_hessian(sum_diff, sum_hess, activation_label);
    end   
end
function is_right = check_grad_loss(cuboid, pixel_loc, intrinsic, extrinsic, ground_truth, plane_ind)
    delta = 0.0000001; check_num = 5; params = generate_cubic_params(cuboid); is_right = 1; judge_cri = 0.1;
    grad_loss = get_grad_loss(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, ground_truth);
    for i = 1 : check_num
        if abs(grad_loss(i)) < judge_cri / 100
            continue;
        end
        cur_params1 = params; cur_params1(i) = cur_params1(i) + delta;
        cur_params2 = params; cur_params2(i) = cur_params2(i) - delta;
        cur_cuboid1 = generate_center_cuboid_by_params(cur_params1);
        cur_cuboid2 = generate_center_cuboid_by_params(cur_params2);
        loss1 = cal_loss_one_hoc(cur_cuboid1, pixel_loc, intrinsic, extrinsic, plane_ind, ground_truth);
        loss2 = cal_loss_one_hoc(cur_cuboid2, pixel_loc, intrinsic, extrinsic, plane_ind, ground_truth);
        num_grad = (loss1 - loss2) / 2 / delta;
        if abs(max(abs(num_grad ./ grad_loss(:, i)) - 1))> judge_cri
            is_right = 0;get_grad_loss
        end
    end
end
function grad_loss = get_grad_loss(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, ground_truth)
    gt = get_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    ft = get_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    grad_gt_theta = get_grad_gt_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    grad_ft_theta = get_grad_ft_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    grad_loss = -2 * (ground_truth - ft * gt) * (grad_ft_theta * gt + ft * grad_gt_theta);
end
function [diff] = cal_loss_one_hoc(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, ground_truth)    
     gt = get_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);     
     ft = get_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);     
     diff = (ground_truth - ft * gt)^2; % diff_true = (ground_truth - gt)^2;
end
function cuboid = generate_center_cuboid_by_params(params)
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
end
function d = get_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind)
    cur_plane_param = get_plane_param(cuboid, plane_ind);
    d = cal_depth_d(cur_plane_param, intrinsic, extrinsic, pixel_loc);
end
function ft = get_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, params)
    cur_plane_param = get_plane_param(cuboid, plane_ind);
    d = cal_depth_d(cur_plane_param, intrinsic, extrinsic, pixel_loc);
    x = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic);
    c = cal_cuboid_corner_point_c(cuboid, plane_ind, params);
    t = calculate_distance_t(c, x, cuboid, plane_ind);
    ft = cal_func_ft(params, t, plane_ind);
end
function grad_gt_theta = get_grad_gt_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, params)
    plane_param = get_plane_param(cuboid, plane_ind);
    ana_grad_a = get_grad_a(cuboid, plane_ind, params);
    ana_grad_d = grad_d(plane_param, pixel_loc, intrinsic, extrinsic);
    grad_gt_theta = ana_grad_d * ana_grad_a;
end
function grad_ft_theta = get_grad_ft_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind, params)
    plane_param = get_plane_param(cuboid, plane_ind);
    ana_grad_a = get_grad_a(cuboid, plane_ind, params);
    ana_grad_d = grad_d(plane_param, pixel_loc, intrinsic, extrinsic);
    grad_d_theta = ana_grad_d * ana_grad_a;
    depth = cal_depth_d(plane_param, intrinsic, extrinsic, pixel_loc);
    g_x_theta = grad_x_theta(pixel_loc, depth, intrinsic, extrinsic, grad_d_theta); x = cal_3d_point_x(pixel_loc, depth, intrinsic, extrinsic);
    g_c_theta = grad_c(params, plane_ind); c = cal_cuboid_corner_point_c(cuboid, plane_ind, params);
    g_t_theta = grad_t(g_c_theta, g_x_theta, c, x, cuboid, plane_ind); t = calculate_distance_t(c, x, cuboid, plane_ind);
    grad_ft_theta = grad_ft(params, g_t_theta, t, plane_ind);
end
function is_right = check_grad_ft(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind)
    delta = 0.0000001; check_num = 5; params = generate_cubic_params(cuboid); is_right = 1; judge_cri = 0.1;
    %{
    plane_param = get_plane_param(cuboid, plane_ind);
    ana_grad_a = get_grad_a(cuboid, plane_ind);
    ana_grad_d = grad_d(plane_param, pixel_loc, intrinsic, extrinsic);
    grad_d_theta = ana_grad_d * ana_grad_a;
    depth = cal_depth_d(plane_param, intrinsic, extrinsic, pixel_loc);
    g_x_theta = grad_x_theta(pixel_loc, depth, intrinsic, extrinsic, grad_d_theta); x = cal_3d_point_x(pixel_loc, depth, intrinsic, extrinsic);
    g_c_theta = grad_c(cuboid, plane_ind); c = cal_cuboid_corner_point_c(cuboid, plane_ind);
    g_t_theta = grad_t(g_c_theta, g_x_theta, c, x, cuboid, plane_ind); t = calculate_distance_t(c, x, cuboid, plane_ind);
    grad_ft_theta = grad_ft(cuboid, g_t_theta, t, plane_ind);
    %}
    grad_ft_theta = get_grad_ft_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    for i = 1 : check_num
        if abs(grad_ft_theta(i)) < judge_cri / 100
            continue;
        end
        cur_params1 = params; cur_params1(i) = cur_params1(i) + delta;
        cur_params2 = params; cur_params2(i) = cur_params2(i) - delta;
        cur_cuboid1 = generate_center_cuboid_by_params(cur_params1);
        cur_cuboid2 = generate_center_cuboid_by_params(cur_params2);
        %{
        cur_plane_param1 = get_plane_param(cur_cuboid1, plane_ind);
        cur_plane_param2 = get_plane_param(cur_cuboid2, plane_ind);
        d1 = cal_depth_d(cur_plane_param1, intrinsic, extrinsic, pixel_loc);
        d2 = cal_depth_d(cur_plane_param2, intrinsic, extrinsic, pixel_loc);
        x1 = cal_3d_point_x(pixel_loc, d1, intrinsic, extrinsic);
        x2 = cal_3d_point_x(pixel_loc, d2, intrinsic, extrinsic);
        c1 = cal_cuboid_corner_point_c(cur_cuboid1, plane_ind);
        c2 = cal_cuboid_corner_point_c(cur_cuboid2, plane_ind);
        t1 = calculate_distance_t(c1, x1, cur_cuboid1, plane_ind);
        t2 = calculate_distance_t(c2, x2, cur_cuboid2, plane_ind);
        ft1 = cal_func_ft(cur_cuboid1, t1, plane_ind);
        ft2 = cal_func_ft(cur_cuboid2, t2, plane_ind);
        %}
        ft1 = get_ft(cur_cuboid1, pixel_loc, intrinsic, extrinsic, plane_ind);
        ft2 = get_ft(cur_cuboid2, pixel_loc, intrinsic, extrinsic, plane_ind);
        num_grad = (ft1 - ft2) / 2 / delta;
        if abs(max(abs(num_grad ./ grad_ft_theta(:, i)) - 1))> judge_cri
            is_right = 0;
        end
    end
end
function is_right = check_grad_gt(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind)
    delta = 0.0000001; check_num = 5; params = generate_cubic_params(cuboid); is_right = 1; judge_cri = 0.001;
    %{
        plane_param = get_plane_param(cuboid, plane_ind);
        ana_grad_a = get_grad_a(cuboid, plane_ind);
        ana_grad_d = grad_d(plane_param, pixel_loc, intrinsic, extrinsic);
        grad_theta = ana_grad_d * ana_grad_a;
    %}
    grad_theta = get_grad_gt_theta(cuboid, pixel_loc, intrinsic, extrinsic, plane_ind);
    for i = 1 : check_num
        if abs(grad_theta(i)) < judge_cri / 100
            continue;
        end
        
        cur_params1 = params; cur_params1(i) = cur_params1(i) + delta;
        cur_params2 = params; cur_params2(i) = cur_params2(i) - delta;
        cur_cuboid1 = generate_center_cuboid_by_params(cur_params1);
        cur_cuboid2 = generate_center_cuboid_by_params(cur_params2);
        %{
            cur_plane_param1 = get_plane_param(cur_cuboid1, plane_ind);
            cur_plane_param2 = get_plane_param(cur_cuboid2, plane_ind);
            d1 = cal_depth_d(cur_plane_param1, intrinsic, extrinsic, pixel_loc);
            d2 = cal_depth_d(cur_plane_param2, intrinsic, extrinsic, pixel_loc);
        %}
        d1 = get_gt(cur_cuboid1, pixel_loc, intrinsic, extrinsic, plane_ind);
        d2 = get_gt(cur_cuboid2, pixel_loc, intrinsic, extrinsic, plane_ind);
        num_grad = (d1 - d2) / 2 / delta;
        if abs(abs(num_grad ./ grad_theta(:, i)) - 1)> judge_cri
            is_right = 0;
        end
    end
end
function grad = get_grad_a(cuboid, plane_ind, params)
    if plane_ind == 1
        grad = get_grad_a1(params);
    end
    if plane_ind == 2
        grad = get_grad_a2(params);
    end
end
function grad = get_grad_a1(params)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    g_theta = [
        -cos(theta);
        -sin(theta);
        0;
        xc * cos(theta) + yc * sin(theta);
        ];
    g_xc = [
        0;
        0;
        0;
        sin(theta);
        ];
    g_yc = [
        0;
        0;
        0;
        -cos(theta);
        ];
    g_l = [
        0;
        0;
        0;
        0;
        ];
    g_w = [
        0;
        0;
        0;
        1/2;
        ];
    g_h = [
        0;
        0;
        0;
        0;
        ];
    grad = [g_theta g_xc g_yc g_l g_w g_h];
end
function grad = get_grad_a2(params)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    g_theta = [
        sin(theta);
        -cos(theta);
        0;
        yc * cos(theta) - xc * sin(theta);
        ];
    g_xc = [
        0;
        0;
        0;
        cos(theta);
        ];
    g_yc = [
        0;
        0;
        0;
        sin(theta);
        ];
    g_l = [
        0;
        0;
        0;
        1/2;
        ];
    g_w = [
        0;
        0;
        0;
        0;
        ];
    g_h = [
        0;
        0;
        0;
        0;
        ];
    grad = [g_theta g_xc g_yc g_l g_w g_h];
end
function check_plane_param(cuboid, plane_ind)
    params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    is_right = true; thresh = 0.0000001; delta_theta = 0 : 0.1 : pi;
    for i = 1 : length(delta_theta)
        params(1) = params(1) + delta_theta(i);
        theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
        cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
        if plane_ind == 1
            param_syn = [-sin(theta), cos(theta), 0, -yc * cos(theta) + xc * sin(theta) + 1/2 * w];
        end
        if plane_ind == 2
            param_syn = [-cos(theta), -sin(theta), 0, yc * sin(theta) + xc * cos(theta) + 1/2 * l];
        end
        plane_num = get_plane_param(cuboid, plane_ind);
        if max(abs(param_syn -plane_num)) > thresh
            is_right = false;
        end
    end
end
function pixel_loc = get_pixel_loc(depth_map, linear_ind)
    [iy, ix] = ind2sub(size(depth_map), linear_ind); pixel_loc = [ix iy];
end
function gt = get_ground_truth(depth_map, linear_ind)
    gt = depth_map(linear_ind);
end
function visualize_3d(extrinsic, intrinsic, cuboid, depth_map, linear_ind_batch, plane_ind_batch)
    figure(1); clf;
    for plane_ind = 1 : 2
        linear_ind = linear_ind_batch(plane_ind_batch == plane_ind);
        % linear_ind = linear_ind_batch;
        if isempty(linear_ind)
            continue;
        end
        plane_param = get_plane_param(cuboid, plane_ind); [iy, ix] = ind2sub(size(depth_map), linear_ind); pixel_loc = [ix iy];
        d = cal_depth_d(plane_param, intrinsic, extrinsic, pixel_loc);
        pts_3d = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic); pts_3d = pts_3d';
        c = cal_cuboid_corner_point_c(cuboid, plane_ind);
        t = calculate_distance_t(c, pts_3d, cuboid, plane_ind);
        ft = cal_func_ft(cuboid, t, plane_ind); cmap = colormap; color_map_ind = map_vector_to_colormap(ft, cmap);
        figure(1); hold on; scatter3(pts_3d(:,1), pts_3d(:,2), pts_3d(:,3), 3, cmap(color_map_ind, :), 'fill');
        hold on; draw_cubic_shape_frame(cuboid); axis equal;
    end
end
function color_map_ind = map_vector_to_colormap(ft, cmap)
    len_cmap = size(cmap,1); ft = ft - min(ft) + 0.0001;
    [~, ind] = sort(ft);
    color_map_ind = ceil(ft / max(ft) * (len_cmap - 2));
end
function grad = grad_loss(ft, gt, A, grad_ft, grad_gt)
    grad = -2 * (A - ft * gt) * (grad_ft * gt + ft * grad_gt);
end
function grad = grad_ft(params, grad_t_theta, t, plane_ind)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    if plane_ind == 1
        [ft, m] = sigmoid_func(t, l);
        grad_cons = [
            0;
            0;
            0;
            - m / (l^2);
            0;
            0;
            ];
        norm_length = l;
        grad_cons = grad_cons';
    end
    if plane_ind == 2
        [ft, m] = sigmoid_func(t, w);
        grad_cons = [
            0;
            0;
            0;
            0;
            - m / (w^2);
            0;
            ];
        norm_length = w;
        grad_cons = grad_cons';
    end
    if ft >= 0
        grad = 2 * ft * (1 - ft) * ( t * grad_cons + m / norm_length * grad_t_theta);
    else
        grad = 1 / 2 * ( t * grad_cons + m / norm_length * grad_t_theta);
    end
end
function grad = grad_c(params, plane_ind)
    if plane_ind == 1
        grad = grad_c1(params);
    end
    if plane_ind == 2
        grad = grad_c2(params);
    end
end
function grad = grad_c2(params)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    g_theta = [
        - 1/2 * l * sin(theta) - 1/2 * w * cos(theta);
        1/2 * l * cos(theta) - 1/2 * w * sin(theta);
        0;
        ];
    g_xc = [
        1;
        0;
        0;
        ];
    g_yc = [
        0;
        1;
        0;
        ];
    g_l = [
        1 / 2 * cos(theta);
        1 / 2 * sin(theta);
        0;
        ];
    g_w = [
        -1/2 * sin(theta);
        1/2 * cos(theta);
        0
        ];
    g_h = [
        0;
        0;
        0;
        ];
    grad = [g_theta g_xc g_yc g_l g_w g_h];
end
function grad = grad_c1(params)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    g_theta = [
        1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta);
        - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta);
        0
        ];
    g_xc = [
        1;
        0;
        0;
        ];
    g_yc = [
        0;
        1;
        0;
        ];
    g_l = [
        - 1 / 2 * cos(theta);
        - 1 / 2 * sin(theta);
        0;
        ];
    g_w = [
        1 / 2 * sin(theta);
        - 1 / 2 * cos(theta);
        0;
        ];
    g_h = [
        0;
        0;
        0;
        ];
    grad = [g_theta g_xc g_yc g_l g_w g_h];
end
function grad = grad_t(gc, gx, c, x, cuboid, plane_ind)
    c1 = c(1); c2 = c(2); x1 = x(1); x2 = x(2);
    gc1 = gc(1, :); gc2 = gc(2, :); gx1 = gx(1, :); gx2 = gx(2, :);
    grad = 1 / 2 * ((c1 - x1)^2 + (c2 - x2)^2)^(-1/2) * ...
        ((2 * (c1 - x1) * (gc1 - gx1)) + 2 * (c2 - x2) * (gc2 - gx2)) * judge_sign(cuboid, x', plane_ind);
end
function grad = grad_x_theta(pixel_loc, depth, intrinsic, extrinsic, gd)
    p1 = pixel_loc(1); p2 = pixel_loc(2); d = depth;
    z = inv(intrinsic * extrinsic);
    z11 = z(1,1); z12 = z(1,2); z13 = z(1,3); z14 = z(1,4);
    z21 = z(2,1); z22 = z(2,2); z23 = z(2,3); z24 = z(2,4);
    z31 = z(3,1); z32 = z(3,2); z33 = z(3,3); z34 = z(3,4);
    z41 = z(4,1); z42 = z(4,2); z43 = z(4,3); z44 = z(4,4);
    grad = [
        z11 * p1 + z12 * p2 + z13;
        z21 * p1 + z22 * p2 + z23;
        z31 * p1 + z32 * p2 + z33;
        ];
    grad = grad * gd;
end
function grad = grad_d(plane_param, pixel_loc, intrinsic, extrinsic)
    z = inv(intrinsic * extrinsic);
    z1 = z(:,1); z2 = z(:,2); z3 = z(:,3); z4 = z(:,4);
    p1 = pixel_loc(1); p2 = pixel_loc(2);
    if size(plane_param, 1) ~= 1
        plane_param = plane_param';
    end
    a = plane_param';
    grad = - z4' * (p1 * a' * z1 + p2 * a' * z2 + a' * z3)^(-1) + ...
        (a' * z4) * (p1 * a' * z1 + p2 * a' * z2 + a' * z3)^(-2) * ...
        (p1 * z1' + p2 * z2' + z3');
end
function diff_inv_record = get_diff_inv(cuboid, pixel_loc_batch, intrinsic, extrinsic, gt_batch, plane_ind_batch)
    params = generate_cubic_params(cuboid);
    diff_inv_record = zeros(size(pixel_loc_batch, 1), 1);
    for plane_ind = 1 : 2
        pixel_loc = pixel_loc_batch(plane_ind_batch == plane_ind, :); gt = gt_batch(plane_ind_batch == plane_ind, :);
        if isempty(gt)
            continue;
        end
        plane_param = get_plane_param(cuboid, plane_ind);
        d = cal_depth_d(plane_param, intrinsic, extrinsic, pixel_loc);
        
        x = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic); x = x';
        c = cal_cuboid_corner_point_c(cuboid, plane_ind, params);
        t = calculate_distance_t(c, x, cuboid, plane_ind);

        ft = cal_func_ft(params, t, plane_ind);
        diff_inv_record(plane_ind_batch == plane_ind) = (gt - ft .* d).^2;
        % cmap = colormap; color_map_ind = map_vector_to_colormap(ft, cmap);
        % figure(2); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(x(:,1),x(:,2),x(:,3),40,cmap(color_map_ind,:),'fill');
    end
end
function plane_param = get_plane_param(cuboid, plane_ind)
    plane_param = cuboid{plane_ind}.params;
end
function d = cal_depth_d(plane_param, intrinsic, extrinsic, pixel_loc)
    z = inv(intrinsic * extrinsic);
    z1 = z(:,1); z2 = z(:,2); z3 = z(:,3); z4 = z(:,4); 
    if size(plane_param, 1) == 1
        plane_param = plane_param';
    end
    a = plane_param; p1 = pixel_loc(:, 1); p2 = pixel_loc(:, 2);
    d = - a' * z4 ./ (p1 * (a' * z1) + p2 * (a' * z2) + a' * z3);
end
function ft = cal_func_ft(params, t, plane_ind)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    if plane_ind == 1
        [ft, ~, t_vals] = sigmoid_func(t, l);
    end
    if plane_ind == 2
        [ft, ~, t_vals] = sigmoid_func(t, w);
    end
    if ft >= 1 / 2
        ft = 2 * (ft - 1/2);
    else
        ft = 1 / 2 * t_vals;
    end
end
function [sig_val, m, t_vals] = sigmoid_func(t, norm_length)
    global sigmoid_m sigmoid_bias
    l = norm_length; th_ = 20; sig_val = zeros(size(t,1), 1);
    m = sigmoid_m; bias = sigmoid_bias; selector = t < th_;
    sig_val(selector) = exp(m / l .* t(selector) + bias) ./ (exp(m / l .* t(selector) + bias) + 1);
    sig_val(~selector) = 1; t_vals = m / l .* t(selector) + bias;
    % sig_val = 2 * (exp(m / l .* t) ./ (exp(m / l .* t) + 1) - 1/2);
end
function t = calculate_distance_t(c, x, cuboid, plane_ind)
    if size(x,1) == 4
        x = x';
    end
    t = sqrt((c(1) - x(:,1)).^2 + (c(2) - x(:,2)).^2) .* judge_sign(cuboid, x, plane_ind);
end
function x = cal_3d_point_x(pixel_loc, d, intrinsic, extrinsic)
    if size(d,1) > 1
        d = d';
    end
    if size(pixel_loc, 2) == 2
        pixel_loc = pixel_loc';
    end
    z = inv(intrinsic * extrinsic);
    z11 = z(1,1); z12 = z(1,2); z13 = z(1,3); z14 = z(1,4);
    z21 = z(2,1); z22 = z(2,2); z23 = z(2,3); z24 = z(2,4);
    z31 = z(3,1); z32 = z(3,2); z33 = z(3,3); z34 = z(3,4);
    z41 = z(4,1); z42 = z(4,2); z43 = z(4,3); z44 = z(4,4);
    p1 = pixel_loc(1,:); p2 = pixel_loc(2,:);
    x = [
        z11 * p1 .* d + z12 * p2 .* d + z13 .* d + z14;
        z21 * p1 .* d + z22 * p2 .* d + z23 .* d + z24;
        z31 * p1 .* d + z32 * p2 .* d + z33 .* d + z34;
        z41 * p1 .* d + z42 * p2 .* d + z43 .* d + z44;
        ];
end
function c = cal_cuboid_corner_point_c(cuboid, plane_ind, params)
    % params = generate_cubic_params(cuboid);
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    if plane_ind == 1
        c = [
            xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta);
            yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta);
            0;
            1;
            ];
    end
    if plane_ind == 2
        c =[
            xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta);
            yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta);
            0;
            1;
            ];
    end
end
function params = generate_cubic_params(cuboid)
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h];
end
%%
function [grad_pos_record, pts3_pos_record] = get_grad_pos(visible_pt_3d, params, extrinsic_param, intrinsic_param, activation_label, depth_map, ratio)
    k1 = visible_pt_3d(:, 1); k2 = visible_pt_3d(:, 2); plane_ind_set = visible_pt_3d(:, 3); pts_num = size(visible_pt_3d, 1);
    grad_pos_record = zeros(size(visible_pt_3d, 1), sum(activation_label)); pts3_pos_record = zeros(size(visible_pt_3d, 1), 3);
    for i = 1 : pts_num
        k = [k1(i) k2(i)]; plane_ind = plane_ind_set(i);
        [grad, diff, pts3] = get_grad_value_and_diff_(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map);
        grad_pos_record(i,:) = grad'; pts3_pos_record(i, :) = pts3;
    end
    pts3_pos_record = [pts3_pos_record ones(size(pts3_pos_record,1),1)];
end
function [A, diff, pts3] = get_grad_value_and_diff_(k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label, depth_map)
    M = intrinsic_param * extrinsic_param;
    pts3 = pts_3d_(params, k, plane_ind)'; pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = M(3, :) * [pts3 1]';
    grad_x_params = get_3d_pt_gradient_(params, k, plane_ind, activation_label);
    grad_img = image_grad_(depth_map, pts2); grad_pixel = pixel_grad_x_(M, pts3); grad_depth = grad_dep_(M, pts3);
    A = (grad_depth * grad_x_params - grad_img * grad_pixel * grad_x_params)';
    % A = (grad_depth * grad_x_params)';
    diff = interpImg(depth_map, pts2) - depth;
    % grad = grad_img * grad_pixel * grad_x_params + grad_depth * grad_x_params;
    %{
    is_right1 = check_grad_depth_params(grad_depth, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    is_right2 = check_grad_pixel_params(grad_pixel, grad_x_params, k, plane_ind, params, extrinsic_param, intrinsic_param, activation_label);
    if ~(is_right1 && is_right2)
        disp('Error')
    end
    %}
end
function pts3 = pts_3d_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(:,1)'; k2 = k(:,2)';
    pts3 = zeros(3, 1);
    if plane_ind == 1
        pts3 = [
            xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * cos(theta) * l;
            yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * sin(theta) * l;
            k2 * h
            ];
    end
    if plane_ind == 2
        pts3 = [
            xc + 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
            yc + 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
            k2 * h
            ];
    end
    if plane_ind == 3
        pts3 = [
            xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
            yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
            k2 * h
            ];
    end
    if plane_ind == 4
        pts3 = [
            xc - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
            yc - 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
            k2 * h
            ];
    end
end
function grad = get_3d_pt_gradient_(params, k, plane_ind, activation_label)
    grad = zeros(3, 6);
    for i = 1 : length(k)
        if activation_label(1)
            grad(:, 1) = g_theta_(params, k, plane_ind);
        end
        if activation_label(2)
            grad(:, 2) = g_xc_(params, k, plane_ind);
        end
        if activation_label(3)
            grad(:, 3) = g_yc_(params, k, plane_ind);
        end
        if activation_label(4)
            grad(:, 4) = g_l_(params, k, plane_ind);
        end
        if activation_label(5)
            grad(:, 5) = g_w_(params, k, plane_ind);
        end
        if activation_label(6)
            grad(:, 6) = g_h_(params, k, plane_ind);
        end
    end
    grad = grad(:, activation_label);
end
function grad = grad_dep_(M, x)
    m3 = M(3, 1:3)';
    grad = m3';
end
function grad = pixel_grad_x_(M, x)
    m1 = M(1, :)'; m2  = M(2, :)'; m3 = M(3, :)';
    if size(x,1) == 1
        x = x';
    end
    if length(x) < 4
        x = [x; 1];
    end
    gx = m1' / (m3' * x) - m3' * (m1' * x) / (m3' * x)^2; gx = gx(1:3);
    gy = m2' / (m3' * x) - m3' * (m2' * x) / (m3' * x)^2; gy = gy(1:3);
    grad = [gx; gy];
end
function grad = image_grad_(image, location)
    step_size = 1;
    x_grad = interpImg(image, [location(1) + step_size, location(2)]) - interpImg(image, [location(1), location(2)]) / step_size;
    y_grad = interpImg(image, [location(1), location(2) + step_size]) - interpImg(image, [location(1), location(2)]) / step_size;
    grad = [x_grad y_grad];
end

function gtheta = g_theta_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gtheta = zeros(3, 1);
    if plane_ind == 1
        gtheta = [
            1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
            -1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * l * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gtheta = [
            -1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
            1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 3
        gtheta = [
            -1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * l * sin(theta);
            1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
            0
            ];
    end
    if plane_ind == 4
        gtheta = [
            1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
            - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
            0
            ];
    end
end
function gxc = g_xc_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gxc = zeros(3, 1);
    if plane_ind == 1
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 2
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 3
        gxc = [
            1;
            0;
            0
            ];
    end
    if plane_ind == 4
        gxc = [
            1;
            0;
            0
            ];
    end
end
function gyc = g_yc_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gyc = zeros(3, 1);
    if plane_ind == 1
        gyc = [
            0;
            1;
            0];
    end
    if plane_ind == 2
        gyc = [
            0;
            1;
            0
            ];
    end
    if plane_ind == 3
        gyc = [
            0;
            1;
            0
            ];
    end
    if plane_ind == 4
        gyc = [
            0;
            1;
            0
            ];
    end
end
function gl = g_l_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gl = zeros(3, 1);
    if plane_ind == 1
        gl = [
            -1 / 2 * cos(theta) + k1 * cos(theta);
            -1 / 2 * sin(theta) + k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 2
        gl = [
            1 / 2 * cos(theta);
            1 / 2 * sin(theta);
            0
            ];
    end
    if plane_ind == 3
        gl = [
            1 / 2 * cos(theta) - k1 * cos(theta);
            1 / 2 * sin(theta) - k1 * sin(theta);
            0
            ];
    end
    if plane_ind == 4
        gl = [
            - 1 / 2 * cos(theta);
            - 1 / 2 * sin(theta);
            0
            ];
    end
end
function gw = g_w_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gw = zeros(3, 1);
    if plane_ind == 1
        gw = [
            1 / 2 * sin(theta);
            - 1 / 2 * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gw = [
            1 / 2 * sin(theta) - k1 * sin(theta);
            -1 / 2 * cos(theta) + k1 * cos(theta);
            0
            ];
    end
    if plane_ind == 3
        gw = [
            -1 / 2 * sin(theta);
            1 / 2 * cos(theta);
            0
            ];
    end
    if plane_ind == 4
        gw = [
            -1 / 2 * sin(theta) + k1 * sin(theta);
            1 / 2 * cos(theta) - k1 * cos(theta);
            0;
            ];
    end
end
function gh = g_h_(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(1); k2 = k(2);
    gh = zeros(3, 1);
    if plane_ind == 1
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 2
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 3
        gh = [
            0;
            0;
            k2
            ];
    end
    if plane_ind == 4
        gh = [
            0;
            0;
            k2
            ];
    end
end
%%
function [diff_record_pos, pts_pos_gt] = get_diff_pos(depth_map, intrinsic_param, extrinsic_param, visible_pt_3d, params, ratio)
    M = intrinsic_param * extrinsic_param; pts3 = zeros(size(visible_pt_3d, 1), 4);
    for i = 1 : size(pts3, 1)
        pts3(i, :) = [(pts_3d__(params, [visible_pt_3d(i, 1) visible_pt_3d(i, 2)], visible_pt_3d(i, 3)))' 1];
    end
    pts2 = project_point_2d(extrinsic_param, intrinsic_param, pts3); depth = (M(3, :) * pts3')';
    gt_depth = zeros(size(pts2, 1), 1);
    for i = 1 : length(gt_depth)
        gt_depth(i) = interpImg(depth_map, pts2(i,:));
    end
    diff_record_pos = (gt_depth - depth).^2;
    projected_pts = [pts2(:,1) .* gt_depth, pts2(:,2) .* gt_depth, gt_depth, ones(size(gt_depth,1),1)]; pts_pos_gt = (inv(intrinsic_param * extrinsic_param) * projected_pts')';
    % figure(1); clf;scatter3(pts_3d_record(:,1),pts_3d_record(:,2),pts_3d_record(:,3),3,'r','fill');hold on;axis equal;
    % draw_cubic_shape_frame(cuboid); hold on; scatter3(pts3(:,1),pts3(:,2),pts3(:,3),3,'g','fill');
    % show_depth_map(depth_map)
end
function pts3 = pts_3d__(params, k, plane_ind)
    theta = params(1); xc = params(2); yc = params(3);
    l = params(4); w = params(5); h = params(6);
    k1 = k(:,1)'; k2 = k(:,2)';
    pts3 = zeros(3, 1);
    if plane_ind == 1
        pts3 = [
            xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k1 * cos(theta) * l;
            yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k1 * sin(theta) * l;
            k2 * h
            ];
    end
    if plane_ind == 2
        pts3 = [
            xc + 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k1 * sin(theta);
            yc + 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k1 * cos(theta);
            k2 * h
            ];
    end
    if plane_ind == 3
        pts3 = [
            xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k1 * l * cos(theta);
            yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k1 * l * sin(theta);
            k2 * h
            ];
    end
    if plane_ind == 4
        pts3 = [
            xc - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k1 * sin(theta);
            yc - 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k1 * cos(theta);
            k2 * h
            ];
    end
end