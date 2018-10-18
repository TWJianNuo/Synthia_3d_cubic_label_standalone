function [sum_diff, sum_hess, sum_loss] = multiple_frame_cubic_estimation(cuboid, intrinsic_param, extrinsic_param, depth_map, linear_ind, visible_pt_3d, activation_label)
    % init
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
    ratio_regularization = 10000000; sigmoid_m = 10; sigmoid_bias = 4;
    sz_depth_map = size(depth_map);
    
    
    [tans_sign_mat1, tans_sign_mat2] = trans_mat_for_sign_judge(theta, xc, yc, l, w, h); % Calculate transformation matrix for sign judgment
    [yy, xx] = ind2sub(sz_depth_map, linear_ind); pixel_loc = [xx yy]; % Get 2d pixel location
    [x, gt, plane_type_rec, sign_inv, selector] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2); % Get 3d pixel location 'x' and depth value 'd'
    pixel_loc = pixel_loc(selector, :); linear_ind = linear_ind(selector); % Organize them so only valide points are included
    tot_num = length(linear_ind); ground_truth = depth_map(linear_ind);
    t = calculate_distance_t(plane_type_rec, x, c1, c2, sign_inv); % calculate distance to the corner points
    ft = cal_func_ft(t, plane_type_rec, l, w, sigmoid_m, sigmoid_bias); % calculate weight value 'ft'
    
    
    % calculate diff term and hessian matrix for inver projection process 
    sum_diff_inv = 0; sum_hess_inv = zeros(sum(activation_label));
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
    end
    loss_inv = sum((ground_truth - gt .* ft).^2);
    
    
    pts3d_pos = pts_3d(visible_pt_3d(:,1:2), visible_pt_3d(:,3), theta, xc, yc, l, w, h)'; % 3d points sampled fromt the cubic shape
    pts2d_pos = (intrinsic_param * extrinsic_param * pts3d_pos')'; depth_pos = pts2d_pos(:, 3);
    pts2d_pos(:,1) = pts2d_pos(:,1) ./ depth_pos; pts2d_pos(:,2) = pts2d_pos(:,2) ./ depth_pos; 
    grad_img = image_grad(depth_map, pts2d_pos); % get image gradient
    diff_pos = interpImg_(depth_map, pts2d_pos(:,1:2)) - depth_pos; loss_pos = sum(diff_pos.^2);
    sum_diff_pos = 0; sum_hess_pos = zeros(sum(activation_label));
    for i = 1 : size(visible_pt_3d, 1)
        grad_x_params = get_3d_pt_gradient([visible_pt_3d(i,1) visible_pt_3d(i,2)], visible_pt_3d(i,3), activation_label, theta, l, w);
        grad_pixel = pixel_grad_x(M, pts3d_pos(i, :)'); grad_depth = M(3, 1:3);
        grad_pos = (grad_depth * grad_x_params - grad_img(i, :) * grad_pixel * grad_x_params)';
        sum_diff_pos = sum_diff_pos + diff_pos(i) * grad_pos';
        sum_hess_pos = sum_hess_pos + grad_pos * grad_pos';
    end
    loss_pos = loss_pos * ratio; sum_diff_pos = sum_diff_pos * ratio; sum_hess_pos = sum_hess_pos * ratio;
    sum_diff = sum_diff_inv + sum_diff_pos; sum_hess = sum_hess_inv + sum_hess_pos + eye(sum(activation_label)) * ratio_regularization; sum_loss = loss_inv + loss_pos;
end
function grad = pixel_grad_x(M, x)
    gx = M(1, :) / (M(3, :) * x) - M(3, :) * (M(1, :) * x) / (M(3, :) * x)^2; gx = gx(1:3);
    gy = M(2, :) / (M(3, :) * x) - M(3, :) * (M(2, :) * x) / (M(3, :) * x)^2; gy = gy(1:3);
    grad = [gx; gy];
end
function grad = get_3d_pt_gradient(k, plane_ind, activation_label, theta, l, w)
    grad = zeros(3, 6);
    for i = 1 : length(k)
        if activation_label(1)
            grad(:, 1) = g_theta(k, plane_ind, theta, l, w);
        end
        if activation_label(2)
            grad(:, 2) = [1 0 0]';
        end
        if activation_label(3)
            grad(:, 3) = [0 1 0]';
        end
        if activation_label(4)
            grad(:, 4) = g_l(k, plane_ind, theta);
        end
        if activation_label(5)
            grad(:, 5) = g_w(k, plane_ind, theta);
        end
        if activation_label(6)
            grad(:, 6) = [0 0 k(2)]';
        end
    end
    grad = grad(:, activation_label);
end
function gw = g_w(k, plane_ind, theta)
    if plane_ind == 1
        gw = [
            1 / 2 * sin(theta);
            - 1 / 2 * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gw = [
            1 / 2 * sin(theta) - k(1) * sin(theta);
            -1 / 2 * cos(theta) + k(1) * cos(theta);
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
            -1 / 2 * sin(theta) + k(1) * sin(theta);
            1 / 2 * cos(theta) - k(1) * cos(theta);
            0;
            ];
    end
end
function gl = g_l(k, plane_ind, theta)
    if plane_ind == 1
        gl = [
            -1 / 2 * cos(theta) +  k(1) * cos(theta);
            -1 / 2 * sin(theta) +  k(1) * sin(theta);
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
            1 / 2 * cos(theta) -  k(1) * cos(theta);
            1 / 2 * sin(theta) -  k(1) * sin(theta);
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
function gtheta = g_theta(k, plane_ind, theta, l, w)
    if plane_ind == 1
        gtheta = [
            1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - k(1) * l * sin(theta);
            -1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) + k(1) * l * cos(theta);
            0
            ];
    end
    if plane_ind == 2
        gtheta = [
            -1 / 2 * l * sin(theta) + 1 / 2 * w * cos(theta) - w * k(1) * cos(theta);
            1 / 2 * l * cos(theta) + 1 / 2 * w * sin(theta) - w * k(1) * sin(theta);
            0
            ];
    end
    if plane_ind == 3
        gtheta = [
            -1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + k(1) * l * sin(theta);
            1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) - k(1) * l * cos(theta);
            0
            ];
    end
    if plane_ind == 4
        gtheta = [
            1 / 2 * l * sin(theta) - 1 / 2 * w * cos(theta) + w * k(1) * cos(theta);
            - 1 / 2 * l * cos(theta) - 1 / 2 * w * sin(theta) + w * k(1) * sin(theta);
            0
            ];
    end
end
function grad = image_grad(image, location)
    x_grad = interpImg_(image, [location(:, 1) + 1, location(:, 2)]) - interpImg_(image, [location(:, 1), location(:, 2)]);
    y_grad = interpImg_(image, [location(:, 1), location(:, 2) + 1]) - interpImg_(image, [location(:, 1), location(:, 2)]);
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
function [x, d, plane_type_rec, sign_inv, selector] = get_plane_and_sign(l, w, pixel_loc, plane_param1, plane_param2, z_mat, tans_sign_mat1, tans_sign_mat2)
    plane_type_rec = zeros(size(pixel_loc,1),1);
    d1 = cal_depth_d(plane_param1', pixel_loc, z_mat); x1 = cal_3d_point_x(pixel_loc, d1, z_mat);
    d2 = cal_depth_d(plane_param2', pixel_loc, z_mat); x2 = cal_3d_point_x(pixel_loc, d2, z_mat);
    x11 = (tans_sign_mat1 * x1')'; x22 = (tans_sign_mat2 * x2')';
    plane_type_rec(x11(:,1) <= l) = 1; plane_type_rec(x22(:,1) < w) = 2;
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