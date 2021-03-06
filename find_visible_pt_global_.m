function visible_pt_label = find_visible_pt_global_(cubics, pts_3d, intrinsic_params, extrinsic_params)
    cuboid = cubics{1}; theta = cuboid{1}.theta; center = mean(cuboid{5}.pts); xc = center(1); yc = center(2); l = cuboid{1}.length1; w = cuboid{2}.length1; h = cuboid{1}.length2;
    num_pt = size(pts_3d, 1); 
    visible_pt_label = false(num_pt, 1);
    % Calculate 2d points
    pts_2d = (intrinsic_params * extrinsic_params * pts_3d')'; 
    pts_2d(:, 1) = pts_2d(:, 1) ./ pts_2d(:, 3); 
    pts_2d(:, 2) = pts_2d(:, 2) ./ pts_2d(:, 3); 
    pts_2d = pts_2d(:, 1:2);
    cam_origin = inv(extrinsic_params) * [0 0 0 1]';
    M = inv(intrinsic_params * extrinsic_params); 
    
    % Calculate affine matrix
    aff = [[1 0 1]', [0 1 1]', [0 0 1]'] * inv([[xc + cos(theta), yc + sin(theta), 1]', [xc - sin(theta), yc + cos(theta), 1]', [xc, yc, 1]']);
    
    % Calculate all possible points
    th = 0.0000000001;
    possible_pts = zeros(4 * num_pt, 4);
    for i = 1 : 4
        params = cuboid{i}.params;
        cur_z = - params * M(:, 4) ./ (params * M(:, 1) * pts_2d(:, 1) + params * M(:, 2) * pts_2d(:, 2) + params * M(:, 3));
        possible_pts((i-1) * num_pt + 1 : i * num_pt, :) = (M * [pts_2d(:,1) .* cur_z, pts_2d(:,2) .* cur_z, cur_z, ones(num_pt, 1)]')';
    end
    
    candidate = (aff * [possible_pts(:,1:2) possible_pts(:,4)]')';
    pre_re = ((abs(abs(candidate(:,1)) - l/2) <  th ) & (abs(candidate(:,2)) < w/2)) | ((abs(abs(candidate(:,2)) - w/2) <  th ) & (abs(candidate(:,1)) < l/2)) | ...
             ((abs(abs(candidate(:,1))) <  th ) & (abs(candidate(:,2)) < w/2)) | ((abs(abs(candidate(:,2))) <  th ) & (abs(candidate(:,1)) < l/2));
    pre_re = pre_re & (possible_pts(:,3) < h);
    dist = ones(num_pt * 4, 1) * inf; dist(pre_re) = sum((possible_pts(pre_re, 1:3) - repmat(cam_origin(1:3)', [sum(pre_re), 1])).^2, 2);
    dist = reshape(dist, [num_pt, 4]); [~, ii] = min(dist, [], 2); selector = (min(dist, [], 2) ~= inf); x_row = 1 : size(dist, 1); linear_ind = sub2ind(size(dist), x_row(selector)', ii(selector));
    visible_pt_label(selector) = sum((possible_pts(linear_ind, 1:3) - pts_3d(selector, 1:3)).^2, 2) < th;
    % figure(1); clf; draw_cubic_shape_frame(cuboid); hold on; scatter3(pts_3d(:,1),pts_3d(:,2),pts_3d(:,3),3,'g','fill')
end