function [sum_diff, sum_hess, sum_loss, dominate_pts, dominate_color] = multiple_frame_cubic_estimation(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pt_3d, activation_label, color1, color2)
    % init
    % load('debug.mat')
    activation_label = (activation_label == 1);
    params = generate_cubic_params(cuboid); 
    theta = params(1);
    xc = params(2);
    yc = params(3);
    l = params(4);
    w = params(5);
    h = params(6);
    [c1, c2] = cal_cuboid_corner_point_c(theta, xc, yc, l, w);
    ratio = 1.5;
    
    z_mat = inv(intrinsic_param * extrinsic_param);
    M = intrinsic_param * extrinsic_param;
    plane_param1 = cuboid{1}.params; plane_param2 = cuboid{2}.params;
    ratio_regularization = 100000; sigmoid_m = 10; sigmoid_bias = 4;
    sz_depth_map = size(depth_map);
    
    
    [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h); % Calculate transformation matrix for sign judgment
    [yy, xx] = ind2sub(sz_depth_map, linear_ind); pixel_loc = [xx yy]; % Get 2d pixel location
    [x, gt, plane_type_rec, sign_inv, selector] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, extrinsic_param); % Get 3d pixel location 'x' and depth value 'd'
    pixel_loc = pixel_loc(selector, :); linear_ind = linear_ind(selector); % Organize them so only valide points are included
    tot_num = length(linear_ind); ground_truth = depth_map(linear_ind);
    t = calculate_distance_t(plane_type_rec, x, c1, c2, sign_inv); % calculate distance to the corner points
    ft = cal_func_ft(t, plane_type_rec, l, w, sigmoid_m, sigmoid_bias); % calculate weight value 'ft'
    
    % calculate diff term and hessian matrix for inver projection process 
    sum_diff_inv = 0; sum_hess_inv = zeros(sum(activation_label));
    grad_inv_record = zeros(tot_num, sum(activation_label));
    for i = 1 : tot_num
        if plane_type_rec(i) == 1
            c = c1;
            plane_param = plane_param1;
        else
            c = c2;
            plane_param = plane_param2;
        end
        ana_grad_a = get_grad_a(theta, xc, yc, plane_type_rec(i)); % gradient of plane_ind
        ana_grad_d = - z_mat(:,4)' * (pixel_loc(i,1) * plane_param * z_mat(:,1) + pixel_loc(i,2) * plane_param * z_mat(:,2) + plane_param * z_mat(:,3))^(-1) + ...
            (plane_param * z_mat(:,4)) * (pixel_loc(i,1) * plane_param * z_mat(:,1) + pixel_loc(i,2) * plane_param * z_mat(:,2) + plane_param * z_mat(:,3))^(-2) * ...
            (pixel_loc(i,1) * z_mat(:,1)' + pixel_loc(i,2) * z_mat(:,2)' + z_mat(:,3)'); % gradient of depth
        grad_gt_theta = ana_grad_d * ana_grad_a;
        grad_temp = [
        z_mat(1,1) * pixel_loc(i,1) + z_mat(1,2) * pixel_loc(i,2) + z_mat(1,3);
        z_mat(2,1) * pixel_loc(i,1) + z_mat(2,2) * pixel_loc(i,2) + z_mat(2,3);
        z_mat(3,1) * pixel_loc(i,1) + z_mat(3,2) * pixel_loc(i,2) + z_mat(3,3);
        ];
        g_x_theta = grad_temp * grad_gt_theta;
        g_c_theta = grad_c(theta, l, w, plane_type_rec(i));
        g_t_theta = 1 / 2 * ((c(1) - x(i,1))^2 + (c(2) - x(i,2))^2)^(-1/2) * ...
                    ((2 * (c(1) - x(i,1)) * (g_c_theta(1, :) - g_x_theta(1, :))) + 2 * (c(2) - x(i,2)) * (g_c_theta(2, :) - g_x_theta(2, :))) * sign_inv(i);
        grad_ft_theta = grad_ft(g_t_theta, t(i), plane_type_rec(i), l, w, sigmoid_m, sigmoid_bias);
        jacob = grad_gt_theta * ft(i) + grad_ft_theta * gt(i); jacob = jacob(activation_label);
        sum_diff_inv = sum_diff_inv + (ground_truth(i) - gt(i) * ft(i)) * jacob; 
        sum_hess_inv = sum_hess_inv + jacob' * jacob;
        
        % For visualization
        grad_inv_record(i,:) = jacob;
        % Do gradient check
        % grad_check_inv(jacob, activation_label, params, plane_type_rec(i), pixel_loc(i,1:2), z_mat, sigmoid_m, sigmoid_bias);
    end
    loss_inv = sum((ground_truth - gt .* ft).^2);
    
    
    pts3d_pos = pts_3d(visible_pt_3d(:,1:2), visible_pt_3d(:,3), theta, xc, yc, l, w, h)'; % 3d points sampled fromt the cubic shape
    pts2d_pos = (intrinsic_param * extrinsic_param * pts3d_pos')'; depth_pos = pts2d_pos(:, 3);
    pts2d_pos(:,1) = pts2d_pos(:,1) ./ depth_pos; pts2d_pos(:,2) = pts2d_pos(:,2) ./ depth_pos; 
    grad_img = image_grad(depth_map, pts2d_pos); % get image gradient
    diff_pos = interpImg_(depth_map, pts2d_pos(:,1:2)) - depth_pos; loss_pos = sum(diff_pos.^2);
    grad_x_params_tot = get_3d_pt_gradient(visible_pt_3d(:,1:2), visible_pt_3d(:,3), theta, l, w);
    sum_diff_pos = 0; sum_hess_pos = zeros(sum(activation_label));
    
    grad_pos_record = zeros(size(visible_pt_3d, 1), sum(activation_label)); % For visualization
    for i = 1 : size(visible_pt_3d, 1)
        grad_x_params = reshape(grad_x_params_tot(:,i,:), [3, 6]);
        grad_pixel = pixel_grad_x(M, pts3d_pos(i, :)'); grad_depth = M(3, 1:3);
        grad_pos = (grad_depth * grad_x_params(:, activation_label) - grad_img(i, :) * grad_pixel * grad_x_params(:, activation_label))';
        sum_diff_pos = sum_diff_pos + diff_pos(i) * grad_pos';
        sum_hess_pos = sum_hess_pos + grad_pos * grad_pos';
        
        % For visualization
        grad_pos_record(i,:) = grad_pos';
        % Do gradient check
        % grad_check_pos(grad_pos, activation_label, params, visible_pt_3d(i, 1:3), intrinsic_param, extrinsic_param, depth_map);
    end
    loss_pos = loss_pos * ratio; sum_diff_pos = sum_diff_pos * ratio; sum_hess_pos = sum_hess_pos * ratio;
    sum_diff = sum_diff_inv + sum_diff_pos; sum_hess = sum_hess_inv + sum_hess_pos + eye(sum(activation_label)) * ratio_regularization; sum_loss = loss_inv + loss_pos;
    
    [flag1, flag2] = judge_flag(nargin);
    if flag1 || flag2
        flag1 = false; flag2 = false;
        if ~flag1 color1 = [zeros(length(linear_ind),1) zeros(length(linear_ind),1) ones(length(linear_ind),1)]; end
        if ~flag2 color2 = [zeros(size(visible_pt_3d, 1),1) ones(size(visible_pt_3d, 1),1) zeros(size(visible_pt_3d, 1),1)]; end
        x_inv_record = x; x_pos_record = pts3d_pos;
        diff_inv_record = (ground_truth - gt .* ft).^2; diff_pos_record = diff_pos.^2;
        pts_3d_gt = calculate_ed_pts(extrinsic_param, intrinsic_param, linear_ind, depth_map(linear_ind), size(depth_map));
        gt_depth = interpImg_(depth_map, pts2d_pos(:,1:2));
        projected_pts = [pts2d_pos(:,1) .* gt_depth, pts2d_pos(:,2) .* gt_depth, gt_depth, ones(size(gt_depth,1),1)]; pts_pos_gt = (inv(intrinsic_param * extrinsic_param) * projected_pts')';
        [dominate_pts, dominate_color] = visualize(cuboid, x_inv_record, x_pos_record, grad_inv_record, grad_pos_record, diff_inv_record, diff_pos_record, visible_pt_3d, intrinsic_param, extrinsic_param, plane_type_rec, pixel_loc, activation_label, pts_3d_gt, pts_pos_gt, color1, color2, l, w);
    end
end
function grad_check_pos(grad_theretical, activation_label, params_, visible_pt_3d, intrinsic_param, extrinsic_param, depth_map)
    delta = 0.00000001; ratio_th = 0.001;
    for j = 1 : length(activation_label)
        if ~activation_label(j)
            continue
        end
        if abs(grad_theretical(j)) < 1e-7
            continue
        end
        for i = 1 : 2
            params = params_;
            if i == 1
                params(j) = params(j) + delta;
            end
            if i == 2
                params(j) = params(j) - delta;
            end
            theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
            pts3d_pos = pts_3d(visible_pt_3d(1:2), visible_pt_3d(3), theta, xc, yc, l, w, h)'; % 3d points sampled fromt the cubic shape
            pts2d_pos = (intrinsic_param * extrinsic_param * pts3d_pos')'; depth_pos = pts2d_pos(3);
            pts2d_pos(1) = pts2d_pos(1) ./ depth_pos; pts2d_pos(2) = pts2d_pos(2) ./ depth_pos;
            if i == 1
                diff_pos1 = interpImg_(depth_map, pts2d_pos(1:2)) - depth_pos;
                % diff_pos1 = depth_pos;
                % val11 = pts2d_pos(1); val12 = pts2d_pos(2);
            end
            if i == 2
                diff_pos2 = interpImg_(depth_map, pts2d_pos(1:2)) - depth_pos;
                % diff_pos2 = depth_pos;
                % val21 = pts2d_pos(1); val122 = pts2d_pos(2);
            end
        end
        ratio = (diff_pos1 - diff_pos2) / 2 / delta / grad_theretical(j);
        if ratio == 0
            continue
        end
        if abs(ratio - 1) > ratio_th
            disp('Error')
        end
    end
end
function grad_check_inv(jacob, activation_label, params_, plane_ind, pixel_loc, z_mat, sigmoid_m, sigmoid_bias)
    delta = 0.00000001; ratio_th = 0.001;
    for j = 1 : length(activation_label)
        if ~activation_label(j)
            continue
        end
        if abs(jacob(j)) < 1e-7
            continue
        end
        if activation_label(j)
            for i = 1 : 2
                params = params_;
                if i == 1
                    params(j) = params(j) + delta;
                end
                if i == 2
                    params(j) = params(j) - delta;
                end
                theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
                cuboid = generate_center_cuboid_by_params(params);
                plane_param1 = cuboid{1}.params; plane_param2 = cuboid{2}.params;
                [c1, c2] = cal_cuboid_corner_point_c(theta, xc, yc, l, w);
                [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h);
                if plane_ind == 1
                    d = cal_depth_d(plane_param1', pixel_loc, z_mat);
                    x = cal_3d_point_x(pixel_loc, d, z_mat);
                    xx = (tans_sign_mat1 * x')';
                    sign_x = sign(xx(1));
                end
                if plane_ind == 2
                    d = cal_depth_d(plane_param2', pixel_loc, z_mat);
                    x = cal_3d_point_x(pixel_loc, d, z_mat);
                    xx = (tans_sign_mat2 * x')';
                    sign_x = sign(xx(1));
                end
                t = calculate_distance_t(plane_ind, x, c1, c2, sign_x); 
                ft = cal_func_ft(t, plane_ind, l, w, sigmoid_m, sigmoid_bias);
                if i == 1
                    val1 = d * ft;
                end
                if i == 2 
                    val2 = d * ft;
                end
            end
            ratio = (val1 - val2) / 2 / delta / jacob(j); 
            if ratio == 0
                continue
            end
            if abs(ratio - 1) > ratio_th
                disp('Error')
            end
        end
    end
end
function cuboid = generate_center_cuboid_by_params(params)
    theta = params(1); xc = params(2); yc = params(3); l = params(4); w = params(5); h = params(6);
    cuboid = generate_cuboid_by_center(xc, yc, theta, l, w, h);
end
function pts_3d = calculate_ed_pts(extrinsic_params, intrinsic_params, linear_ind, depth, sz_depth)
    [yy, xx] = ind2sub([sz_depth(1) sz_depth(2)], linear_ind); pixel_2d = [xx yy]; % lin_check = sub2ind(sz_depth, pixel_2d(:,2), pixel_2d(:,1));
    tmp = [pixel_2d(:,1) .* depth, pixel_2d(:,2) .* depth, depth, ones(size(depth,1),1)];
    pts_3d = (inv(intrinsic_params * extrinsic_params) * tmp')';
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
function [dominate_pts, dominate_color] = visualize(cuboid, x_inv_record, x_pos_record, grad_inv_record, grad_pos_record, diff_inv_record, diff_pos_record, visible_pt_3d, intrinsic_param, extrinsic_param, plane_ind_batch, pixel_loc_batch, activation_label, pts_3d_gt, pts_pos_gt, color1, color2, l, w)
    min_rad = min([l, w]); min_rad = min_rad(1) / 5; if min_rad < 3, min_rad = 3; end
    max_ratio = 10; params_org = generate_cubic_params(cuboid); params_org_ = params_org(activation_label);
    grad_inv_record = grad_inv_record / norm(grad_inv_record) / max_ratio;
    grad_pos_record = grad_pos_record / norm(grad_pos_record) / max_ratio;
    
    params1 = repmat(params_org, [size(grad_inv_record, 1) 1]); params2 = repmat(params_org, [size(grad_pos_record, 1) 1]);
    params1(:, activation_label) = params1(:, activation_label) + grad_inv_record; 
    params2(:, activation_label) = params2(:, activation_label) + grad_pos_record;
    
    pts3_inv_change = zeros(size(grad_inv_record, 1), 4);
    pts3_pos_change = zeros(size(grad_pos_record, 1), 4);
    
    z_mat = inv(intrinsic_param * extrinsic_param);
    for i = 1 : size(pts3_inv_change, 1)
        params = params1(i,:);
        cur_cuboid = generate_center_cuboid_by_params(params);
        theta = params(1);
        xc = params(2);
        yc = params(3);
        l = params(4);
        w = params(5);
        h = params(6);
        plane_param1 = cur_cuboid{1}.params; plane_param2 = cur_cuboid{2}.params;
        if plane_ind_batch(i) == 1
            d_ = cal_depth_d(plane_param1', pixel_loc_batch(i,1:2), z_mat);
        else
            d_ = cal_depth_d(plane_param2', pixel_loc_batch(i,1:2), z_mat);
        end
        pts3_inv_change(i,:) = cal_3d_point_x(pixel_loc_batch(i,1:2), d_, z_mat);
    end
    
    for i = 1 : size(pts3_pos_change, 1)
        k = [visible_pt_3d(i,1) visible_pt_3d(i,2)]; plane_ind = visible_pt_3d(i,3);
        params = params2(i,:);
        theta = params(1);
        xc = params(2);
        yc = params(3);
        l = params(4);
        w = params(5);
        h = params(6);
        pts3_pos_change(i,:) = pts_3d(k, plane_ind, theta, xc, yc, l, w, h)';
    end
    
    dir_inv = pts3_inv_change(:,1:3) - x_inv_record(:,1:3); dir_inv = dir_inv ./ repmat(vecnorm(dir_inv, 2, 2), [1 3]);
    dir_pos = pts3_pos_change(:,1:3) - x_pos_record(:,1:3); dir_pos = dir_pos ./ repmat(vecnorm(dir_pos, 2, 2), [1 3]);
    
    val = [diff_inv_record; diff_pos_record]; colors = generate_cmap_array(val); pts = [x_inv_record; x_pos_record];
    quiv_size = (val - min(val)); quiv_size = quiv_size / max(quiv_size) * 3 + 0.5; 
    quiv_size_inv = quiv_size(1: size(dir_inv,1)); quiv_size_pos = quiv_size(size(dir_inv,1) + 1 : end);
    
    dominate_pts = [x_inv_record; x_pos_record]; dominate_selector = (quiv_size > 1); dominate_pts = dominate_pts(dominate_selector, :);
    dominate_color = [color1; color2]; dominate_color = dominate_color(dominate_selector, :);
    figure(1); clf;
    draw_cubic_shape_frame(cuboid); hold on;
    scatter3(x_inv_record(:,1),x_inv_record(:,2),x_inv_record(:,3),min_rad,color1,'fill'); hold on;
    pts_c = [pts_pos_gt; pts_3d_gt];
    % pts_c = pts_3d_gt;
    scatter3(pts_c(:,1),pts_c(:,2),pts_c(:,3),min_rad,'c','fill'); hold on;
    scatter3(x_pos_record(:,1),x_pos_record(:,2),x_pos_record(:,3),min_rad,color2,'fill'); hold on;
    for i = 1 : size(x_inv_record,1)
        % quiver3(x_inv_record(i,1),x_inv_record(i,2),x_inv_record(i,3),dir_inv(i,1),dir_inv(i,2),dir_inv(i,3),quiv_size_inv(i),'b'); hold on;
    end
    for i = 1 : size(x_pos_record,1)
        % quiver3(x_pos_record(i,1),x_pos_record(i,2),x_pos_record(i,3),dir_pos(i,1),dir_pos(i,2),dir_pos(i,3),quiv_size_pos(i),'g'); hold on;
    end
    for i = 1 : size(x_inv_record,1)
        % plot3([x_inv_record(i,1);pts_3d_gt(i,1)],[x_inv_record(i,2);pts_3d_gt(i,2)],[x_inv_record(i,3);pts_3d_gt(i,3)],'LineStyle',':','Color','k'); hold on;
    end
    for i = 1 : size(x_pos_record,1)
        % plot3([x_pos_record(i,1);pts_pos_gt(i,1)],[x_pos_record(i,2);pts_pos_gt(i,2)],[x_pos_record(i,3);pts_pos_gt(i,3)],'LineStyle',':','Color','k'); hold on;
    end
end
function colors = generate_cmap_array(val)
    cmap = colormap(); val = val - min(val) + 0.1;
    colors = cmap(ceil(val / max(val) * (size(cmap,1) - 1)), :);
end
function grad = pixel_grad_x(M, x)
    gx = M(1, :) / (M(3, :) * x) - M(3, :) * (M(1, :) * x) / (M(3, :) * x)^2; gx = gx(1:3);
    gy = M(2, :) / (M(3, :) * x) - M(3, :) * (M(2, :) * x) / (M(3, :) * x)^2; gy = gy(1:3);
    grad = [gx; gy];
end
function grad = get_3d_pt_gradient(k, plane_ind, theta, l, w)
    tot_num = length(plane_ind);
    grad = cat(3, g_theta(k, plane_ind, theta, l, w), repmat([1;0;0], [1, tot_num]), repmat([0;1;0], [1, tot_num]), g_l(k, plane_ind, theta), g_w(k, plane_ind, theta), [zeros(1,tot_num);zeros(1,tot_num);k(:,2)']);
end
function gw = g_w(k, plane_ind, theta)
    gw = zeros(3, length(plane_ind));
    for cur_pd = 1 : length(plane_ind)
        selector = (plane_ind == cur_pd);
        if cur_pd == 1
            gw(:, selector) = [
                1 / 2 * sin(theta) * ones(1,sum(selector));
                - 1 / 2 * cos(theta) * ones(1,sum(selector));
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 2
            gw(:, selector) = [
                1 / 2 * sin(theta) - k(selector,1)' * sin(theta);
                -1 / 2 * cos(theta) + k(selector,1)' * cos(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 3
            gw(:, selector) = [
                -1 / 2 * sin(theta) * ones(1,sum(selector));
                1 / 2 * cos(theta) * ones(1,sum(selector));
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 4
            gw(:, selector) = [
                -1 / 2 * sin(theta) + k(selector,1)' * sin(theta);
                1 / 2 * cos(theta) - k(selector,1)' * cos(theta);
                zeros(1,sum(selector));
                ];
        end
    end
end
function gl = g_l(k, plane_ind, theta)
    gl = zeros(3, length(plane_ind));
    for cur_pd = 1 : length(plane_ind)
        selector = (plane_ind == cur_pd);
        if cur_pd == 1
            gl(:, selector) = [
                -1 / 2 * cos(theta) +  k(selector, 1)' * cos(theta);
                -1 / 2 * sin(theta) +  k(selector, 1)' * sin(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 2
            gl(:, selector) = [
                1 / 2 * cos(theta) * ones(1,sum(selector));
                1 / 2 * sin(theta) * ones(1,sum(selector));
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 3
            gl(:, selector) = [
                1 / 2 * cos(theta) -  k(selector, 1)' * cos(theta);
                1 / 2 * sin(theta) -  k(selector, 1)' * sin(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 4
            gl(:, selector) = [
                - 1 / 2 * cos(theta) * ones(1,sum(selector));
                - 1 / 2 * sin(theta) * ones(1,sum(selector));
                zeros(1,sum(selector));
                ];
        end
    end
end
function gtheta = g_theta(k, plane_ind, theta, l, w)
    gtheta = zeros(3, length(plane_ind));
    for cur_pd = 1 : length(plane_ind)
        selector = (plane_ind == cur_pd);
        if cur_pd == 1
            gtheta(:, selector) = [
                1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k(selector,1)' * l * sin(theta);
                -1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k(selector,1)' * l * cos(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 2
            gtheta(:, selector) = [
                -1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k(selector,1)' * cos(theta);
                1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k(selector,1)' * sin(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 3
            gtheta(:, selector) = [
                -1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k(selector,1)' * l * sin(theta);
                1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k(selector,1)' * l * cos(theta);
                zeros(1,sum(selector));
                ];
        end
        if cur_pd == 4
            gtheta(:, selector) = [
                1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k(selector,1)' * cos(theta);
                - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k(selector,1)' * sin(theta);
                zeros(1,sum(selector));
                ];
        end
    end
end
function grad = image_grad(image, location)
    step = 1e-7;
    x_grad = interpImg_(image, [location(:, 1) + step / 2, location(:, 2)]) - interpImg_(image, [location(:, 1) - step / 2, location(:, 2)]); x_grad = x_grad / step;
    y_grad = interpImg_(image, [location(:, 1), location(:, 2) + step / 2]) - interpImg_(image, [location(:, 1), location(:, 2)] - step / 2); y_grad = y_grad / step;
    % x_grad = interpImg_(image, [location(:, 1) + step, location(:, 2)]) - interpImg_(image, [location(:, 1), location(:, 2)]); x_grad = x_grad / step;
    % y_grad = interpImg_(image, [location(:, 1), location(:, 2) + step]) - interpImg_(image, [location(:, 1), location(:, 2)]); y_grad = y_grad / step;
    grad = [x_grad y_grad];
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
function grad = grad_ft(grad_t_theta, t, plane_ind, l, w, sigmoid_m, sigmoid_bias)
    if plane_ind == 1
        ft = sigmoid_func(t, l, sigmoid_m, sigmoid_bias);
        grad_cons = [
            0;
            0;
            0;
            - sigmoid_m / (l^2);
            0;
            0;
            ];
        norm_length = l;
        grad_cons = grad_cons';
    end
    if plane_ind == 2
        ft = sigmoid_func(t, w, sigmoid_m, sigmoid_bias);
        grad_cons = [
            0;
            0;
            0;
            0;
            - sigmoid_m / (w^2);
            0;
            ];
        norm_length = w;
        grad_cons = grad_cons';
    end
    if ft >= 0
        grad = 2 * ft * (1 - ft) * ( t * grad_cons + sigmoid_m / norm_length * grad_t_theta);
    else
        grad = 1 / 2 * ( t * grad_cons + sigmoid_m / norm_length * grad_t_theta);
    end
end
function grad = grad_c(theta, l, w, plane_ind)
    if plane_ind == 1
        grad = grad_c1(theta, l, w);
    end
    if plane_ind == 2
        grad = grad_c2(theta, l, w);
    end
end
function grad = grad_c2(theta, l, w)
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
function grad = grad_c1(theta, l, w)
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
function grad = get_grad_a(theta, xc, yc, plane_ind)
    if plane_ind == 1
        grad = get_grad_a1(theta, xc, yc);
    end
    if plane_ind == 2
        grad = get_grad_a2(theta, xc, yc);
    end
end
function grad = get_grad_a1(theta, xc, yc)
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
function grad = get_grad_a2(theta, xc, yc)
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
function ft = cal_func_ft(t, plane_type_rec, l, w, sigmoid_m, sigmoid_bias)
    selector = (plane_type_rec == 1);
    ft = zeros(size(t)); t_vals = zeros(size(t));
    [ft(selector), t_vals(selector)] = sigmoid_func(t(selector), l, sigmoid_m, sigmoid_bias);
    [ft(~selector), t_vals(~selector)] = sigmoid_func(t(~selector), w, sigmoid_m, sigmoid_bias);
    
    selector = (ft >= 1/2); 
    num_ans1 = 2 * (ft - 1/2); 
    num_ans2 = 1 / 2 * t_vals;
    ft(selector) = num_ans1(selector); 
    ft(~selector) = num_ans2(~selector);
end
function [sig_val, t_vals] = sigmoid_func(t, norm_length, sigmoid_m, sigmoid_bias)
    th_ = 20; sig_val = zeros(size(t,1), 1);
    selector = t < th_;
    sig_val(selector) = exp(sigmoid_m / norm_length .* t(selector) + sigmoid_bias) ./ (exp(sigmoid_m / norm_length .* t(selector) + sigmoid_bias) + 1);
    sig_val(~selector) = 1; 
    t_vals = sigmoid_m / norm_length .* t + sigmoid_bias;
end
function t = calculate_distance_t(plane_type_rec, x, c1, c2, sign_inv)
    selector = (plane_type_rec == 1);
    t = zeros(size(x,1),1);
    t(selector) = sqrt((c1(1) - x(selector,1)).^2 + (c1(2) - x(selector,2)).^2) .* sign_inv(selector);
    t(~selector) = sqrt((c2(1) - x(~selector,1)).^2 + (c2(2) - x(~selector,2)).^2) .* sign_inv(~selector);
end
function [c1, c2] = cal_cuboid_corner_point_c(theta, xc, yc, l, w)
    c1 = [
        xc - 1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta);
        yc - 1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta);
        0;
        1;
        ];
    c2 = [
        xc + 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta);
        yc + 1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta);
        0;
        1;
        ];
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
function d = cal_depth_d(plane_param, pixel_loc, z_mat)
    d = - plane_param' * z_mat(:,4) ./ (pixel_loc(:,1) * (plane_param' * z_mat(:,1)) + pixel_loc(:,2) * (plane_param' * z_mat(:,2)) + plane_param' * z_mat(:,3));
end
function [x, d, plane_type_rec, sign_inv, selector] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2, extrinsic)
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
    
    sign_1 = sign(x11(:,1)); sign_2 = sign(x22(:,1));
    selector = (plane_type_rec ~= 0); selector1 = (plane_type_rec == 1); selector2 = (plane_type_rec == 2);
    d = zeros(size(d1)); d(selector1) = d1(selector1); d(selector2) = d2(selector2); d = d(selector);
    x = zeros(size(x1)); x(selector1, : ) = x1(selector1, :); x(selector2, :) = x2(selector2, :); x = x(selector, :);
    sign_inv = zeros(size(sign_1)); sign_inv(selector1) = sign_1(selector1); sign_inv(selector2) = sign_2(selector2); sign_inv = sign_inv(selector);
    plane_type_rec = plane_type_rec(selector);
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
function params = generate_cubic_params(cuboid)
    theta = cuboid{1}.theta; l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2);
    params = [theta, xc, yc, l, w, h];
end