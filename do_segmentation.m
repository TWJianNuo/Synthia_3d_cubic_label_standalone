function do_segmentation(helper)
    for i = 1 : length(helper)
        [base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, n, save_path, inter_path] = read_helper_info(helper, i);
        prev_mark = cell(0); max_instance = 1;
        for frame = 1 : n
            [extrinsic_params, intrinsic_params, depth, label, instance, rgb, affine_matrx, align_matrix] = grab_provided_data(base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, frame, inter_path);
            [max_instance, prev_mark, plot_figure] = seg_image(depth, label, instance, prev_mark, extrinsic_params, intrinsic_params, affine_matrx, max_instance, frame, align_matrix, rgb, helper{i});
            output_results(prev_mark, plot_figure, frame, rgb, extrinsic_params, helper{i})
        end
    end
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
function output_results(prev_mark, plot_figure, frame, rgb, extrinsic_params, helper_entry)
    %{
    if ~isempty(plot_figure) 
        imwrite(plot_figure, [exp_re_path '/align_situation_' num2str(frame) '.png']); 
    end
    
    plot_mark(prev_mark, extrinsic_params); 
    F = getframe(gcf); 
    [X, ~] = frame2im(F); 
    imwrite(X, [exp_re_path '/3d_pts_' num2str(frame) '.png']);
    %}
    path1 = [helper_entry{9} '/segmentation_re']; path2 = [helper_entry{9} '/seg_raw_re_sv'];
    if frame == 1
        mkdir(path1); mkdir(path2)
    end
    new_img = render_image(prev_mark, rgb); 
    imwrite(new_img, [path1 '/_instance_label' num2str(frame) '.png']);
    
    save_intance(prev_mark, frame, path2)
end
function save_intance(prev_mark, frame, folder_path)
    full_path = [folder_path '/' num2str(frame, '%06d') '.mat'];
    save(full_path, 'prev_mark');
end
function intrinsic_params = read_intrinsic(base_path)
    txtPath = [base_path 'CameraParams/' 'intrinsics.txt'];
    vec = load(txtPath);
    focal = vec(1); cx = vec(2); cy = vec(3);
    intrinsic_params = [focal, 0, cx; 0, focal, cy; 0, 0, 1]; intrinsic_params(4,4) = 1;
end
function [extrinsic_params, intrinsic_params, depth, label, instance, rgb, affine_matrix, align_matrix] = grab_provided_data(base_path, GT_Depth_path, GT_seg_path, GT_RGB_path, GT_Color_Label_path, cam_para_path, frame, inter_path)
    intrinsic_params = read_intrinsic(base_path);
    f = num2str(frame, '%06d');
    txtPath = strcat(base_path, cam_para_path, num2str((frame-1), '%06d'), '.txt'); vec = load(txtPath); extrinsic_params = reshape(vec, 4, 4);
    ImagePath = strcat(base_path, GT_Depth_path, f, '.png'); depth = getDepth(ImagePath);
    ImagePath = strcat(base_path, GT_seg_path, f, '.png'); [label, instance] = getIDs(ImagePath);
    ImagePath = strcat(base_path, GT_RGB_path, f, '.png'); rgb = imread(ImagePath);
    load([inter_path '/affine_matrix.mat']); affine_matrix = reshape(affine_matrix_recorder(frame, :), [4 4]);
    load([inter_path '/adjust_matrix.mat']);
    if (frame > 1)
        align_matrix = reshape(param_record(frame - 1, :),[4,4]);
    else
        align_matrix = zeros(4,4);
    end
end
function path = make_dir()
    father_folder = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/';
    DateString = datestr(datetime('now'));
    DateString = strrep(DateString,'-','_');DateString = strrep(DateString,' ','_');DateString = strrep(DateString,':','_'); DateString = DateString(1:14);
    path = [father_folder DateString '_segmentation'];
    mkdir(path);
end
function new_img = render_image(prev_mark, rgb)
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    for i = 1 : length(prev_mark)
        color = ceil(prev_mark{i}.color * 254);
        ind = prev_mark{i}.linear_ind;
        r(ind) = color(1); g(ind) = color(2); b(ind) = color(3);
    end
    new_img = uint8(zeros(size(rgb))); new_img(:,:,1) = r; new_img(:,:,2) = g; new_img(:,:,3) = b;
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
function [min_obj_pixel_num, min_height, max_height, kernel_size, map_ration] = read_and_translate_sup_mat(help_info_entry, frame)
    path = [help_info_entry{9} 'seg_sup_mat.mat'];
    load(path);
    min_obj_pixel_num = seg_sup_mat{frame}.min_obj_pixel_num;
    min_height = seg_sup_mat{frame}.min_height;
    max_height = seg_sup_mat{frame}.max_height;
    kernel_size = seg_sup_mat{frame}.kernel_size;
    map_ration = seg_sup_mat{frame}.map_ration;
end
function [max_instance, prev_mark, plot_figure] = seg_image(depth_map, label, instance, prev_mark, extrinsic_params, intrinsic_params, affine_matrx, max_instance, frame, align_matrix, rgb, help_info_entry)
    plot_figure = zeros(0); candidate_objs = [0, 2, 5, 6, 7, 9, 13, 14, 15];
    % min_obj_pixel_num = [200, 1000, 200, 20, 20, inf, 200, 200, inf];
    % min_height = [0.1, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf];
    % max_height = [inf, inf, inf, 1, 4, inf, inf, inf, inf];
    % kernel_size = [2, 3, 6, 6, 6, 3, 3, 3, 3];
    % image_size = [60 60]; 
    [min_obj_pixel_num, min_height, max_height, kernel_size, map_ration_] = read_and_translate_sup_mat(help_info_entry, frame);
    mark_stack = cell(0); mark_bck = prev_mark;
    for jj = 1 : length(candidate_objs)
    % for jj = 5 : 5
        cur_mark = cell(100, 1); obj_num_count = 1;
        obj_type = candidate_objs(jj); obj_type_ind = find(candidate_objs == obj_type);
        [ix, iy] = find(label == obj_type); SE = strel('square',kernel_size(obj_type_ind)); 
        % [ix, iy] = find(label == building_type); down_sample_rate = 10;
        % ix = ix(1:down_sample_rate:end); iy = iy(1:down_sample_rate:end);
        linear_ind_record = sub2ind(size(depth_map), ix, iy);
        cur_old_pts_set = get_3d_pts(depth_map, extrinsic_params, intrinsic_params, linear_ind_record);
        cur_new_pts_set = (affine_matrx * (cur_old_pts_set)')'; 
        
        selectors = (cur_new_pts_set(:,3) > min_height(obj_type_ind)) & (cur_new_pts_set(:,3) < max_height(obj_type_ind));
        linear_ind_record = linear_ind_record(selectors); ix = ix(selectors); iy = iy(selectors);
        cur_old_pts_set = cur_old_pts_set(selectors, :);
        cur_new_pts_set = cur_new_pts_set(selectors, :); 
        if length(linear_ind_record) < 20
            continue;
        end
        map_ration = map_ration_(obj_type_ind);
        l = [max(ix) - min(ix), max(iy) - min(iy)]; l = ceil(max(l) * map_ration); 
        if obj_type == 0
            a = 1;
        end
        image_size = [l, l]; bimg = false(image_size);
        prev_mark = find_co_type_prev_mark(mark_bck, obj_type);
        if isempty(prev_mark)
            bimg_linear_ind = ind2d(cur_new_pts_set(:, 1:2), image_size); bimg(bimg_linear_ind) = true; J = imdilate(bimg,SE); J = imerode(J,SE); CC = bwconncomp(J);
            for i = 1 : CC.NumObjects
                now_search_b_ind = CC.PixelIdxList{i}; % now_found_bindices = zeros(0);
                is_member_re = ismember(bimg_linear_ind, now_search_b_ind);
                now_found_bindices = find(is_member_re);
                % for j = 1 : length(now_search_b_ind)
                %     now_found_bindices = [now_found_bindices; find(bimg_linear_ind == now_search_b_ind(j))];
                % end
                if length(now_found_bindices) > min_obj_pixel_num(obj_type_ind)
                    [cur_mark{obj_num_count}, max_instance] = init_mark(max_instance, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), rand([1 3]), cur_new_pts_set(now_found_bindices, :), frame, align_matrix, obj_type);
                    obj_num_count = obj_num_count + 1;
                end
            end
        else
            prev_new_pts_set = zeros(0); prev_old_pts_set = zeros(0); pre_new_pts = zeros(0); integrated_instance_set = zeros(0); integrated_color_set = zeros(0);
            for i = 1 : length(prev_mark)
                prev_new_pts_set = [prev_new_pts_set; prev_mark{i}.pts_new];
                prev_old_pts_set = [prev_old_pts_set; prev_mark{i}.pts_old];
                integrated_instance_set = [integrated_instance_set; prev_mark{i}.instanceId * ones(size(prev_mark{i}.pts_old,1),1)];
                integrated_color_set = [integrated_color_set; repmat(prev_mark{i}.color, [size(prev_mark{i}.pts_old,1),1])];
            end
            integrated_instance_set = [integrated_instance_set; zeros(size(cur_old_pts_set, 1), 1)];
            trans_cur_old_pts_set = (align_matrix * cur_old_pts_set')'; trans_cur_new_pts_set =  (prev_mark{i}.affine_matrx * trans_cur_old_pts_set')';
            integrated_new_pts_set = [prev_new_pts_set; trans_cur_new_pts_set]; start_ind_of_cur_frame = size(prev_new_pts_set, 1);
            integrated_bimg_linear_ind = ind2d(integrated_new_pts_set(:, 1:2), image_size);
            prev_bimg_linear_ind = integrated_bimg_linear_ind(1 : size(prev_new_pts_set, 1)); cur_bimg_linear_ind = integrated_bimg_linear_ind(size(pre_new_pts, 1) + 1 : end);
            bimg(integrated_bimg_linear_ind) = true; J = imdilate(bimg,SE); J = imerode(J,SE); CC = bwconncomp(J);
            cur_mark = cell(CC.NumObjects, 1);
            
            % plot_figure = generate_figure_image(trans_cur_new_pts_set, prev_new_pts_set, integrated_color_set, J);
            for i = 1 : CC.NumObjects
                now_search_b_ind = CC.PixelIdxList{i}; % now_found_bindices = zeros(0);
                is_member_re = ismember(integrated_bimg_linear_ind, now_search_b_ind);
                now_found_bindices = find(is_member_re);
                % for j = 1 : length(now_search_b_ind)
                %     now_found_bindices = [now_found_bindices; find(integrated_bimg_linear_ind == now_search_b_ind(j))];
                % end
                % figure(1); clf; scatter3(integrated_new_pts_set(now_found_bindices,1),integrated_new_pts_set(now_found_bindices,2),integrated_new_pts_set(now_found_bindices,3),3,integrated_color_set(now_found_bindices,:),'fill');
                if length(now_found_bindices) > min_obj_pixel_num(obj_type_ind)
                    % output_split_procedure(frame, J, now_search_b_ind, prev_new_pts_set(now_found_bindices(now_found_bindices <= start_ind_of_cur_frame), :), trans_cur_new_pts_set(now_found_bindices(now_found_bindices > start_ind_of_cur_frame) - start_ind_of_cur_frame, :), integrated_color_set(now_found_bindices(now_found_bindices <= start_ind_of_cur_frame), :), i, prev_new_pts_set, trans_cur_new_pts_set)
                    unique_found_instance_id = unique(integrated_instance_set(now_found_bindices));
                    now_found_bindices = now_found_bindices(now_found_bindices > start_ind_of_cur_frame) - start_ind_of_cur_frame;
                    if length(unique_found_instance_id) == 1 && unique_found_instance_id(1) == 0
                        [cur_mark{obj_num_count}, max_instance] = init_mark(max_instance, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), rand([1 3]), cur_new_pts_set(now_found_bindices, :),frame, align_matrix, obj_type);
                    else
                        father_instanceId = min(unique_found_instance_id(unique_found_instance_id~=0));
                        if (isempty(now_found_bindices)) continue; end
                        % figure(1); clf; scatter3(trans_cur_new_pts_set(now_found_bindices,1),trans_cur_new_pts_set(now_found_bindices,2),trans_cur_new_pts_set(now_found_bindices,3),3,'r','fill');
                        [cur_mark{obj_num_count}, ~] = init_mark(father_instanceId, linear_ind_record(now_found_bindices), extrinsic_params, intrinsic_params, affine_matrx, cur_old_pts_set(now_found_bindices, :), find_color(father_instanceId, prev_mark), cur_new_pts_set(now_found_bindices, :), frame, align_matrix, obj_type);
                    end
                    obj_num_count = obj_num_count + 1;
                end
            end
        end
        indices = find(~cellfun('isempty', cur_mark)); cur_mark = cur_mark(indices);
        mark_stack = [mark_stack; cur_mark];
    end
    
    instance_clust = unique(instance); bias = 100000;
    for i = 1 : length(instance_clust)
        if instance_clust(i) == 0
            continue;
        end
        cur_instance = instance_clust(i);
        [yy, xx] = find(instance == cur_instance);
        linear_ind = sub2ind(size(depth_map), yy, xx); depth_val = depth_map(linear_ind);
        if length(linear_ind) < 200
            continue;
        end
        projected_pts = [xx .* depth_val, yy .* depth_val, depth_val, ones(length(depth_val), 1)];
        pts_old = (inv(intrinsic_params * extrinsic_params) * projected_pts')';
        pts_new = ((affine_matrx * pts_old'))';
        obj_type = label(linear_ind(ceil(end / 2)));
        [cur_mark, ~] = init_mark(cur_instance + bias, linear_ind, extrinsic_params, intrinsic_params, affine_matrx, pts_old, find_color(cur_instance + bias, mark_bck), pts_new, frame, align_matrix, obj_type);
        mark_stack = [mark_stack; cur_mark];
    end
    
    
    
    % rgb_new = visualize_mark(mark_stack, rgb); figure(1); clf; imshow(rgb_new)
    prev_mark = mark_stack; % plot_mark(cur_mark);
end
function [filtered_mark, selector] = find_co_type_prev_mark(prev_mark, obj_ind)
    selector = false(length(prev_mark), 1);
    for i = 1 : length(prev_mark)
        if prev_mark{i}.obj_type == obj_ind
            selector(i) = true;
        end
    end
    filtered_mark = prev_mark(selector);
end
function rgb_new = visualize_mark(cur_mark, rgb)
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    for i = 1 : length(cur_mark)
        pts_new = cur_mark{i}.pts_new;
        affine = cur_mark{i}.affine_matrx;
        extrinsic_param = cur_mark{i}.extrinsic_params;
        intrinsic_param = cur_mark{i}.intrinsic_params;
        color = cur_mark{i}.color; color = uint8(ceil(color * 255));
        [pts2d, depth] = project_point_2d(extrinsic_param, intrinsic_param, pts_new, affine); pts2d = round(pts2d);
        linear_ind = sub2ind(size(r), pts2d(:,2), pts2d(:,1));
        r(linear_ind) = color(1);
        g(linear_ind) = color(2);
        b(linear_ind) = color(3);
    end
    rgb_new = cat(3, r, g, b);
end
function output_split_procedure(frame, J, now_search_b_ind, prev_new_pts_set, trans_cur_new_pts_set, integrated_color_set, index, prev, cur)
    show_num = 28; path = '/home/ray/ShengjieZhu/Fall Semester/depth_detection_project/Exp_re/segmentation_results/18_Sep_2018_21_segmentation/Frame28_steps/';
    if show_num == frame
        r = 255 * uint8(J); g = 255 * uint8(J); b = 255 * uint8(J);
        g(now_search_b_ind) = 0; b(now_search_b_ind) = 0;
        f = figure('visible', 'off'); clf; scatter3(prev_new_pts_set(:,1),prev_new_pts_set(:,2),prev_new_pts_set(:,3),10,integrated_color_set,'fill');
        hold on; scatter3(trans_cur_new_pts_set(:,1),trans_cur_new_pts_set(:,2),trans_cur_new_pts_set(:,3),6,'c','fill'); 
        hold on; scatter3(prev(:,1),prev(:,2),prev(:,3),3,'r','fill'); hold on; scatter3(cur(:,1),cur(:,2),cur(:,3),3,'g','fill'); axis equal;
        F = getframe(f); [X, ~] = frame2im(F); imwrite(X, [path 'split_procedure_3dpoints_' num2str(frame) '_' num2str(index) '.png'])
        color_binary_img = uint8(zeros([size(J), 3])); color_binary_img(:,:,1) = r; color_binary_img(:,:,2) = g; color_binary_img(:,:,3) = b;
        imwrite(color_binary_img, [path 'split_procedure_binary_map' num2str(frame) '_' num2str(index) '.png'])
    end
end
function X = generate_figure_image(trans_cur_new_pts_set, prev_new_pts_set, integrated_color_set, J)
    f = figure('visible','off'); 
    % f = figure(1); clf;
    clf; scatter3(trans_cur_new_pts_set(:,1),trans_cur_new_pts_set(:,2),trans_cur_new_pts_set(:,3),3,'r','fill');
    hold on; scatter3(prev_new_pts_set(:,1),prev_new_pts_set(:,2),prev_new_pts_set(:,3),5,integrated_color_set,'fill');
    % figure(2); clf; imshow(fliplr(J));
    F = getframe(f); [X, ~] = frame2im(F);
end
function [cur_mark, max_instance] = init_mark(max_instance, linear_ind, extrinsic_params, intrinsic_params, affine_matrx, pts_old, color, pts_new, frame, adjust_matrix, obj_type)
    cur_mark.linear_ind = linear_ind; cur_mark.instanceId = max_instance;
    cur_mark.extrinsic_params = extrinsic_params; cur_mark.intrinsic_params = intrinsic_params;
    cur_mark.affine_matrx = affine_matrx; 
    cur_mark.adjust_matrix = adjust_matrix;
    cur_mark.pts_old = pts_old;
    cur_mark.color = color; cur_mark.pts_new = pts_new;
    cur_mark.frame = frame;
    cur_mark.obj_type = obj_type;
    max_instance = max_instance + 1;
end
function color = find_color(instanceId, prev)
    color = zeros(1,3);
    for i = 1 : length(prev)
        if prev{i}.instanceId == instanceId
            color = prev{i}.color;
        end
    end
    if isequal(color, zeros(1,3))
        color = rand(1,3);
    end
end
function plot_mark(cur_mark, extrinsic_params)
    figure('visible', 'off')
    clf;
    for i = 1 : length(cur_mark)
        pts = cur_mark{i}.pts_new;
        scatter3(pts(:,1),pts(:,2),pts(:,3),3,cur_mark{i}.color,'fill'); hold on;
    end
    % [camera_position, camera_direction] = get_camera_pos_and_direction(extrinsic_params);
    % scatter3(camera_position(1), camera_position(2), camera_position(3), 10, 'b', 'fill'); hold on;
    % quiver3(camera_position(1), camera_position(2), camera_position(3), camera_direction(1), camera_direction(2), camera_direction(3), 15);
    axis equal
end
function [camera_position, camera_dirction] = get_camera_pos_and_direction(extrinsic_params)
    logic_origin = [0 0 0 1]; logic_positive_direction = [0 0 1 1];
    camera_position = (inv(extrinsic_params) * logic_origin')';
    camera_dirction = (inv(extrinsic_params) * logic_positive_direction')'; camera_dirction = camera_dirction - camera_position;
end
function bimg_linear_ind = ind2d(pts2d, image_size)
    xmin = min(pts2d(:,1)); xmax = max(pts2d(:,1)); ymin = min(pts2d(:,2)); ymax = max(pts2d(:,2));
    rangex = xmax - xmin; rangey = ymax - ymin;
    pixel_coordinate_x = (pts2d(:,1) - xmin) / (rangex / (image_size(1) - 1)); pixel_coordinate_x = round(pixel_coordinate_x) + 1;
    pixel_coordinate_y = (pts2d(:,2) - ymin) / (rangey / (image_size(2) - 1)); pixel_coordinate_y = round(pixel_coordinate_y) + 1;
    bimg_linear_ind = sub2ind(image_size, pixel_coordinate_y, pixel_coordinate_x);
end