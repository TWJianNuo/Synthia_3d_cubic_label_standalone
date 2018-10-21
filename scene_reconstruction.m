help_info = helper();
% estimate_affine_matrix(help_info);
% align_frame(help_info)
% generate_seg_mat(help_info)
% do_segmentation(help_info);
% create_rgb_cluster(help_info);
% union_and_save_all_marks(help_info);
% organize_data_structure_for_multiple_frames(help_info);
% generate_depth_cluster(help_info)
% do_multiple_frame_based_optimization(help_info)
cubic_shape_setimation_on_multiple_frame(help_info);
% check_close_obj_effects(help_info);
function estimate_affine_matrix(helper)
    for i = 1 : length(helper)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, n, save_path, inter_path] = read_helper_info(helper, i);
        affine_matrix_recorder = zeros(n, 16); error_recorder = zeros(n, 1);
        for frame = 1 : n
            [extrinsic_params, intrinsic_params, depth, label, instance, rgb] = grab_provided_data(base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, frame);
            [affine_matrx, mean_error] = estimate_ground_plane(intrinsic_params, extrinsic_params, depth, label, helper{i}, frame);
            affine_matrix_recorder(frame, :) = affine_matrx(:);
            error_recorder(frame) = mean_error;
        end
        save([inter_path 'affine_matrix' ],'affine_matrix_recorder')
    end
end
function print_camera_param(frame_num, fileID, intrinsic, extrinsic, affine_matrix)
    if frame_num == 1
        print_matrix(fileID, intrinsic);
    end
    fprintf(fileID, '%d\n', frame_num);
    print_matrix(fileID, extrinsic);
    print_matrix(fileID, affine_matrix);
end
function file_ids = create_txt(folder_path)
    file_ids = struct();
    file_ids.txt_cubic = fopen([folder_path '/' 'cubic_info.txt'], 'w');
    file_ids.txt_camera = fopen([folder_path '/' 'camera_info.txt'], 'w');
end
function save_work(file_ids, objs, frame, img, rgb, folder_path, intrinsic, extrinsic, affine_matrix)
    print_cubic_info(file_ids.txt_cubic, objs, frame);
    print_camera_param(frame, file_ids.txt_camera, intrinsic, extrinsic, affine_matrix)
    imwrite(rgb, [folder_path '/raw_img/' num2str(frame) '.png']);
    imwrite(img, [folder_path '/rendered_img/' num2str(frame) '.png']);
end
function folder_path = make_dir(save_path, base_path)
    C = strsplit(base_path,'/'); sub_name = C{5};
    folder_path = [save_path sub_name];
    mkdir(folder_path); mkdir([folder_path '/raw_img']); mkdir([folder_path '/rendered_img']); 
end
function print_cubic_info(file_id, objs, frame_id)
    fprintf(file_id, '%d\t%d\n', frame_id, length(objs));
    for i = 1 : length(objs)
        print_matrix(file_id, objs{i}.guess)
    end
end
function print_matrix(file_id, matrix)
    for i = 1 : size(matrix, 1)
        for j = 1 : size(matrix, 2)
            fprintf(file_id, '%5d\t', matrix(i, j));
        end
        fprintf(file_id, '\n');
    end
end
function img = render_image(objs, img)
    for i = 1 : length(objs)
        img = cubic_lines_of_2d(img, objs{i}.cur_cuboid, objs{i}.intrinsic_params, objs{i}.extrinsic_params);
    end
end
function draw_cars(objs)
    for i = 1 : length(objs)
        new_pts = objs{i}.new_pts;
        hold on; scatter3(new_pts(:,1),new_pts(:,2),new_pts(:,3),3,'r','fill');
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
function save_img(img, frame)
    path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/SYNTHIA-SEQS-05-SPRING/Car_reconstruction_results/';
    f = num2str(frame, '%06d');
    imwrite(img, [path f '.png']);
end
function save_mean_to_text(ave_dist, ave_diff, path)
    fileID = fopen(path,'a');
    fprintf(fileID,'Average Distance:\t%5d\n',ave_dist);
    fprintf(fileID,'Average Difference:\t%5d\n',ave_diff);
    fclose(fileID);
end
function save_to_text(objs, frame_num, path)
    fileID = fopen(path,'a');
    for i = 1 : length(objs)
        fprintf(fileID,'Frame_num:\t%2d\n',frame_num);
        fprintf(fileID,'3D Distance:\t%5d\n',objs{i}.metric(1));
        fprintf(fileID,'Depth Difference:\t%5d\n',objs{i}.metric(2));
    end
    fclose(fileID);
end
function extrinsic_params = get_new_extrinsic_params(extrinsic_params, affine_matrix)
    extrinsic_params = extrinsic_params / affine_matrix;
end
function objs = estimate_single_cubic_shape(objs, extrinsic_params, intrinsic_params, index, affine_matrix)
    % Iinit:
    extrinsic_params = get_new_extrinsic_params(extrinsic_params, affine_matrix);
    R = extrinsic_params(1:3, 1:3);
    T = extrinsic_params(1:3, 4);
    image_size = size(objs{1}.depth_map);
    camera_origin = (-R' * T)';
    num1 = 10; num2 = 10;
    gamma = 0.5; delta_threshold = 0.01;
    activation_label = [1 1 1 1 1 0];
    it_count = 0;
    is_terminated = false; terminate_ratio = 0.05; distortion_terminate_ratio = 1.1;
    max_it_num = 300;
    tot_dist_record = zeros(max_it_num, 1); tot_params_record = zeros(max_it_num, 6);
    
    tot_dist_record(1) = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_params_record(1, :) = objs{index}.guess;
    while ~is_terminated
        it_count = it_count + 1;
        cur_activation_label = cancel_co_activation_label(activation_label);
        
        cubics = distill_all_eisting_cubic_shapes(objs);
        % cubics = {objs{index}.cur_cuboid};
        cur_pts = sample_cubic_by_num(objs{index}.cur_cuboid, num1, num2);
        [pts_estimated_2d, pts_estimated_vlaid, ~, depth] = projectPoints(cur_pts, intrinsic_params(1:3, 1:3), extrinsic_params, [0,0,0,0,0], [image_size(1) image_size(2)], false);
        cur_pts = cur_pts(pts_estimated_vlaid, :); pts_estimated_2d = pts_estimated_2d(pts_estimated_vlaid, :); depth = depth(pts_estimated_vlaid);
        [visible_pt_3d, ~, ~] = find_visible_pt_global(cubics, pts_estimated_2d, cur_pts, depth, intrinsic_params, extrinsic_params, camera_origin);
        objs{index}.visible_pt = visible_pt_3d;
        
        activated_params_num = sum(double(cur_activation_label));
        hessian = zeros(activated_params_num, activated_params_num); first_order = zeros(activated_params_num, 1);
        [hessian, first_order] = analytical_gradient(objs{index}.cur_cuboid, intrinsic_params, extrinsic_params, visible_pt_3d, objs{index}.depth_map, hessian, first_order, cur_activation_label);
        
        
        figure(1)
        clf
        scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
        hold on
        draw_cubic_shape_frame(cubics{index});
        hold on
        scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'k', 'fill')
        axis equal; view(-46.3, 23.6)
        % F = getframe(gcf);
        % [X, Map] = frame2im(F);
        % imwrite(X, ['/home/ray/Desktop/exp_new/' num2str(it_count) '.png'])
        
        [delta, terminate_flag_singular] = calculate_delta(hessian, first_order); [params_cuboid_order, terminate_flag] = update_params(objs{index}.guess, delta, gamma, cur_activation_label, terminate_ratio);
        objs{index}.guess(1:6) = params_cuboid_order;
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
        ave_dist = calculate_ave_distance(objs{index}.cur_cuboid, objs{index}.new_pts); tot_dist_record(it_count + 1) = ave_dist; tot_params_record(it_count + 1, :) = objs{index}.guess;
        
        % if max(abs(delta)) < delta_threshold || it_count >= max_it_num || terminate_flag || terminate_flag_singular || (tot_dist_record(it_count + 1) / min(tot_dist_record(1:(it_count + 1)))) > distortion_terminate_ratio
        %     is_terminated = true;
        % end
    end
    objs = find_best_fit_cubic(objs, tot_dist_record, tot_params_record, index);
    %{
    figure(1)
    clf
    scatter3(visible_pt_3d(:,1), visible_pt_3d(:,2), visible_pt_3d(:,3), 3, 'r', 'fill');
    hold on
    draw_cuboid(cubics{index});
    hold on
    scatter3(objs{index}.new_pts(:,1), objs{index}.new_pts(:,2), objs{index}.new_pts(:,3), 3, 'g', 'fill')
    %}
end
function objs = find_best_fit_cubic(objs, tot_dist_record, tot_params_record, index)
    selector = (tot_dist_record ~= 0); tot_dist_record = tot_dist_record(selector); tot_params_record = tot_params_record(selector, :);
    min_index = find(tot_dist_record == min(tot_dist_record));
    if length(min_index) > 1
        warning('Multiple minimum values')
        min_index = min_index(1);
    end
    params_cuboid_order = tot_params_record(min_index, :);
    try
        cx = params_cuboid_order(1); cy = params_cuboid_order(2); theta = params_cuboid_order(3); l = params_cuboid_order(4); w = params_cuboid_order(5); h = params_cuboid_order(6);
        objs{index}.guess = tot_params_record(min_index, :);
        objs{index}.cur_cuboid = generate_cuboid_by_center(cx, cy, theta, l, w, h);
    catch
        disp('Error occurred')
    end
end

function [delta, terminate_flag] = calculate_delta(hessian, first_order)
    lastwarn(''); % Empty existing warning
    delta = hessian \ first_order;
    % delta = (hessian + eye(size(hessian,1))) \ first_order;
    [msgstr, msgid] = lastwarn;
    terminate_flag = false;
    if strcmp(msgstr,'矩阵为奇异工作精度。') && strcmp(msgid, 'MATLAB:singularMatrix')
        delta = 0;
        disp('Frame Discarded due to singular Matrix, terminated')
        terminate_flag = true;
    end
end
function cubics = distill_all_eisting_cubic_shapes(objs)
    cubics = cell(length(objs), 1);
    for i = 1 : length(objs)
        cubics{i} = objs{i}.cur_cuboid;
    end
end
function objs = get_init_guess(objs)
    for i = 1 : length(objs)
        [params, cuboid] = estimate_rectangular(objs{i}.new_pts);
        objs{i}.guess = params;
        objs{i}.cur_cuboid = cuboid;
    end
end
function objs = seg_image(depth_map, label, instance, extrinsic_params, intrinsic_params, affine_matrix)
    % Only for car currently;
    car_label = 8;
    max_depth = max(max(depth_map));
    
    tot_obj_num = 0;
    objs = cell(tot_obj_num);
    
    existing_instance = unique(instance);
    labelled_pixel = false(size(instance));
    
    for i = 1 : length(existing_instance)
        cur_instance = existing_instance(i);
        if cur_instance == 0
            continue;
        end
        [ix, iy] = find(instance == cur_instance);
        linear_ind = sub2ind(size(instance), ix, iy); selector = (label(linear_ind) == car_label); linear_ind = linear_ind(selector);
        if length(linear_ind) < 10
            continue
        end
        labelled_pixel(linear_ind) = true;
        
        type = label(linear_ind(1));
        instance_id = instance(linear_ind(1));
        tot_obj_num = tot_obj_num + 1;
        objs{tot_obj_num, 1} = init_single_obj(depth_map, linear_ind, extrinsic_params, intrinsic_params, type, instance_id, max_depth, affine_matrix);
        instance(linear_ind) = 0;
        label(linear_ind) = 0;
    end
end
function metric = calculate_metric(obj)
    % One term is the distance from the points to the cubic shape
    % The other term is the differences between the depth
    pts = obj.new_pts;
    cuboid = obj.cur_cuboid;
    ave_dist = calculate_ave_distance(cuboid, pts);
    ave_diff = calculate_depth_diff(obj.depth_map, obj.visible_pt, obj.extrinsic_params, obj.intrinsic_params);
    metric = [ave_dist ave_diff];
end
function ave_diff = calculate_depth_diff(depth_map, pts, extrinsic_params, intrinsic_params)
    [pts2d, depth] = get_2dloc_and_depth(pts, extrinsic_params, intrinsic_params, size(depth_map));
    linear_ind = sub2ind(size(depth_map), pts2d(:,2), pts2d(:,1)); gt_depth = depth_map(linear_ind);
    selector = (gt_depth ~= max(gt_depth)); ave_diff = sum(abs(depth(selector) - gt_depth(selector))) / sum(selector);
    
    % Code to check:
    depth_map_copy = depth_map; linear_ind = sub2ind(size(depth_map), pts2d(:,2), pts2d(:,1));
    depth_map_copy(linear_ind) = depth;
    %{
    figure(4)
    clf
    show_depth_map(depth_map_copy);
    %}
end
function [pts2d, depth] = get_2dloc_and_depth(pts, extrinsic_params, intrinsic_params, img_size)
    pts2d = (intrinsic_params * extrinsic_params * [pts(:, 1:3) ones(size(pts,1),1)]')';
    depth = pts2d(:,3);
    pts2d(:,1) = pts2d(:,1) ./ depth; pts2d(:,2) = pts2d(:,2) ./ depth;
    pts2d = round(pts2d(:,1:2));
    selector = (pts2d <= 0); pts2d(selector) = 1;
    selector = (pts2d(:,1) > img_size(2)); pts2d(selector, 1) = img_size(2);
    selector = (pts2d(:,2) > img_size(1)); pts2d(selector, 2) = img_size(1);
end
function objs = further_seg(extrinsic_params, intrinsic_params, image_size, objs, ind, SE, min_pixel_num)
    pts_3d = objs{ind}.new_pts;
    xmin = min(pts_3d(:,1)); xmax = max(pts_3d(:,1)); ymin = min(pts_3d(:,2)); ymax = max(pts_3d(:,2));
    rangex = xmax - xmin; rangey = ymax - ymin;
    bimg = false(image_size);
    pixel_coordinate_x = (pts_3d(:,1) - xmin) / (rangex / (image_size(1) - 1)); pixel_coordinate_x = round(pixel_coordinate_x) + 1;
    pixel_coordinate_y = (pts_3d(:,2) - ymin) / (rangey / (image_size(2) - 1)); pixel_coordinate_y = round(pixel_coordinate_y) + 1;
    linear_ind = sub2ind(image_size, pixel_coordinate_y, pixel_coordinate_x);
    bimg(linear_ind) = true;
    J = imdilate(bimg,SE); J = imerode(J,SE);
    CC = bwconncomp(J);
    
    depth_map = objs{ind}.depth_map; type = objs{ind}.type; max_depth = max(max(depth_map)); tot_linear_ind = objs{ind}.linear_ind;
    objs(ind) = [];
    for i = 1 : CC.NumObjects
        img_ind = CC.PixelIdxList{i};
        cur_indices = zeros(0);
        for j = 1 : length(img_ind)
            cur_indices = [cur_indices; find(linear_ind == img_ind(j))];
        end
        if length(cur_indices) > min_pixel_num
            cur_linear_ind = tot_linear_ind(cur_indices);
            objs{end + 1, 1} = init_single_obj(depth_map, cur_linear_ind, extrinsic_params, intrinsic_params, type, 0, max_depth);
        end
    end
end
function label = eliminate_type_pixel(label, min_obj_height, max_obj_height, extrinsic_params, intrinsic_params, depth_map, type)
    min_height = min_obj_height(type); max_height = max_obj_height(type);
    [ix, iy] = find(label == type);
    linear_ind = sub2ind(size(label), ix, iy);
    old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind);
    new_pts = get_pt_on_new_coordinate_system(old_pts);
    selector = (new_pts(:, 3) < min_height) | (new_pts(:, 3) > max_height);
    label(linear_ind(selector)) = 0;
end

function obj = init_single_obj(depth_map, linear_ind, extrinsic_params, intrinsic_params, type, instance_id, max_depth, affine_matrix)
    obj = struct;
    obj.instance = instance_id;
    obj.type = type;
    
    obj.linear_ind = linear_ind;
    obj.depth_map = ones(size(depth_map)) * max_depth;
    obj.depth_map(obj.linear_ind) = depth_map(obj.linear_ind);
    
    obj.old_pts = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, obj.linear_ind);
    obj.new_pts = (affine_matrix * obj.old_pts')';
    obj.extrinsic_params = get_new_extrinsic_params(extrinsic_params, affine_matrix); obj.intrinsic_params = intrinsic_params;
end

function draw_scene(objs, fig_index, color_gt)
    cmap = colormap;
    new_color_gt = uint8(zeros(size(color_gt)));
    figure(fig_index)
    clf
    for i = 1 : length(objs)
        color = cmap(randi([1 64]), :);
        pts = objs{i}.new_pts;
        [I,J] = ind2sub(size(objs{i}.depth_map),objs{i}.linear_ind);
        for k = 1 : length(objs{i}.linear_ind)
            color_gt(I(k), J(k), :) = uint8([256 256 256]);
            new_color_gt(I(k), J(k), :) = uint8(round(color * 255));
        end
        scatter3(pts(:,1), pts(:,2), pts(:,3), 3, color, 'fill')
        hold on
        draw_cuboid(objs{i}.cur_cuboid)
    end
    axis equal
    figure(2)
    imshow(color_gt)
    figure(3)
    imshow(new_color_gt)
end
function reconstructed_3d = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, valuable_ind)
    height = size(depth_map, 1);
    width = size(depth_map, 2);
    x = 1 : height; y = 1 : width;
    [X, Y] = meshgrid(y, x);
    pts = [Y(:) X(:)];
    projects_pts = [pts(valuable_ind,2) .* depth_map(valuable_ind), pts(valuable_ind,1) .* depth_map(valuable_ind), depth_map(valuable_ind), ones(length(valuable_ind), 1)];
    reconstructed_3d = (inv(intrinsic_params * extrinsic_params) * projects_pts')';
end
function show_depth_map(depth_map)
    imshow(uint16(depth_map * 1000));
end
function [params_cuboid_order, terminate_flag] = update_params(old_params, delta, gamma, activation_label, termination_ratio)
    terminate_flag = false;
    activation_label = (activation_label == 1);
    new_params = old_params;
    params_derivation_order = [new_params(3), new_params(1), new_params(2), new_params(4), new_params(5), new_params(6)];
    if max(abs(delta ./ params_derivation_order(activation_label))) < termination_ratio
        terminate_flag = true;
    end
    params_derivation_order(activation_label) = params_derivation_order(activation_label) + gamma * delta';
    params_cuboid_order = [params_derivation_order(2), params_derivation_order(3), params_derivation_order(1), params_derivation_order(4), params_derivation_order(5), params_derivation_order(6)];
    if params_cuboid_order(4) < 0 || params_cuboid_order(5) < 0 || params_cuboid_order(6) < 0
        params_cuboid_order = old_params;
        terminate_flag = true;
        disp('Impossible cubic shape, terminated')
    end
end
function activation_label = cancel_co_activation_label(activation_label)
    activation_label = (activation_label == 1);
    if (activation_label(2) || activation_label(3)) && (activation_label(5) || activation_label(4))
        if(randi([1 2], 1) == 1)
            activation_label(2) = 0; activation_label(3) = 0;
        else
            activation_label(5) = 0; activation_label(4) = 0;
        end
    end
end
function img = cubic_lines_of_2d(img, cubic, intrinsic_params, extrinsic_params)
    % color = uint8(randi([1 255], [1 3]));
    % color = rand([1 3]);
    shapeInserter = vision.ShapeInserter('Shape', 'Lines', 'BorderColor', 'White');
    pts3d = zeros(8,4);
    for i = 1 : 4
        pts3d(i, :) = [cubic{i}.pts(1, :) 1];
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
        % figure(1)
        % imshow(img)
        % pause()
    end
end
