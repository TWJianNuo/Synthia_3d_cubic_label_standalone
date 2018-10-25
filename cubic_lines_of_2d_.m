function linear_ind_tot = cubic_lines_of_2d_(l, w, cubic, intrinsic_params, extrinsic_params, label, linear_ind_org)
    % color = uint8(randi([1 255], [1 3]));
    % color = rand([1 3]);
    linear_ind_tot = zeros(0);
    sky_ind = 1; Road_ind = 3; Sidewalk_ind = 4; Lanemarking_ind = 12; 
    permitted_ind = [sky_ind Road_ind Sidewalk_ind Lanemarking_ind];
    pts3d = zeros(8,4);
    try
        mh = cubic{1}.mh;
    catch
        mh = 0;
    end
    for i = 1 : 4
        pts3d(i, :) = [cubic{i}.pts(1, :) 1];
        pts3d(i, 3) = pts3d(i, 3) + mh;
    end
    for i = 5 : 8
        pts3d(i, :) = [cubic{5}.pts(i - 4, :) 1];
    end
    pts_3d_cam_cor = (extrinsic_params * [pts3d(:, 1:3) ones(size(pts3d,1),1)]')';
    if min(pts_3d_cam_cor(:,3)) < 0.2
        return
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
    x1 = lines(:,1); y1 = lines(:,2); x2 = lines(:,3); y2 = lines(:,4);
    cross_pts = [...
        ones(12,1), - x1 .* (y2 - y1) ./ (x2 - x1) + y1, ...
        ones(12,1) * l, (y2 - y1) ./ (x2 - x1) .* (l - x1) + y1, ...
        -y1 .* (x2 - x1) ./ (y2 - y1) + x1, ones(12,1), ...
        (x2 - x1) ./ (y2 - y1) .* (w - y1) + x1, ones(12,1) * w ...
        ];

    selector1 = [
        cross_pts(:,1) >=1 & cross_pts(:,1) <= l & cross_pts(:,2) >= 1 & cross_pts(:,2) <= w, ...
        cross_pts(:,3) >=1 & cross_pts(:,3) <= l & cross_pts(:,4) >= 1 & cross_pts(:,4) <= w, ...
        cross_pts(:,5) >=1 & cross_pts(:,5) <= l & cross_pts(:,6) >= 1 & cross_pts(:,6) <= w, ...
        cross_pts(:,7) >=1 & cross_pts(:,7) <= l & cross_pts(:,8) >= 1 & cross_pts(:,8) <= w, ...
        ];
    selector2 = [
        lines(:,1) >= 1 & lines(:,1) <= l & lines(:,2) >= 1 & lines(:,2) <= w, ...
        lines(:,3) >= 1 & lines(:,3) <= l & lines(:,4) >= 1 & lines(:,4) <= w, ...
        ];
    selector2 = ~selector2;
    for i = 1 : 12
        for j = 1 : 2
            if selector2(i,j)
                ind = find(selector1(i,:));
                if isempty(ind)
                    lines(i, 2*j - 1 : 2*j) = [1 1];
                    continue
                end
                pts_out = [lines(i, 2*j - 1), lines(i, 2 * j)];
                pts1 = [cross_pts(i, 2*ind(1) - 1), cross_pts(i, 2*ind(1))];
                pts2 = [cross_pts(i, 2*ind(2) - 1), cross_pts(i, 2*ind(2))];
                d1 = norm(pts_out - pts1); d2 = norm(pts_out - pts2);
                if d1 < d2
                    lines(i, 2*j - 1 : 2*j) = pts1;
                else
                    lines(i, 2*j - 1 : 2*j) = pts2;
                end
            end
        end
    end
    
    for i = 1 : 12
        leng1 = abs(lines(i,1) - lines(i,3));
        leng2 = abs(lines(i,2) - lines(i,4));
        if leng1 > leng2
            leng = leng1;
        else
            leng = leng2;
        end
        xx = round(linspace(lines(i,1), lines(i,3), leng));
        yy = round(linspace(lines(i,2), lines(i,4), leng));
        linear_ind = sub2ind([w, l], yy, xx);
        selector = ismember(label(linear_ind), permitted_ind) | ismember(linear_ind, linear_ind_org);
        linear_ind_tot = [linear_ind_tot linear_ind(selector)];
    end
    %{
    % lines(xx2, yy2 * 2 - 1) = cross_pts(xx1, yy1 * 2 - 1);
    % lines(xx2, yy2 * 2) = cross_pts(xx1, yy1 * 2);
    for i = 9 : 9
        figure(1); clf; scatter(lines(i,1),lines(i,2),10,'r','fill');
        hold on; scatter(lines(i,3),lines(i,4),10,'r','fill');
        hold on; scatter(cross_pts(i,1),cross_pts(i,2),10,'b','fill');
        hold on; scatter(cross_pts(i,3),cross_pts(i,4),10,'b','fill');
        hold on; scatter(cross_pts(i,5),cross_pts(i,6),10,'b','fill');
        hold on; scatter(cross_pts(i,7),cross_pts(i,8),10,'b','fill');
        axis equal
    end
    selector = [
        ((cross_pts(:, 1) - x1) .* (cross_pts(:, 1) - x2) < 0) ...
        & ((cross_pts(:, 2) - y1) .* (cross_pts(:, 2) - y2) < 0), ...
        ((cross_pts(:, 3) - x1) .* (cross_pts(:, 3) - x2) < 0) ...
        & ((cross_pts(:, 4) - y1) .* (cross_pts(:, 4) - y2) < 0), ...
        ((cross_pts(:, 5) - x1) .* (cross_pts(:, 5) - x2) < 0) ...
        & ((cross_pts(:, 6) - y1) .* (cross_pts(:, 6) - y2) < 0), ...
        ((cross_pts(:, 7) - x1) .* (cross_pts(:, 7) - x2) < 0) ...
        & ((cross_pts(:, 8) - y1) .* (cross_pts(:, 8) - y2) < 0), ...
        ];
    [xx, yy] = find(selector);
    if ~isempty(xx)
        selector1 = x1 < 0 | x1 > l | y1 < 0 | y1 > w;
        selector2 = x2 < 0 | x2 > l | y2 < 0 | y2 > w;
        lines(selector1, 1) = cross_pts(selector1, yy * 2 - 1);
        lines(selector1, 2) = cross_pts(selector1, yy * 2);
        lines(selector2, 1) = cross_pts(selector2, yy * 2 - 1);
        lines(selector2, 1) = cross_pts(selector2, yy * 2 - 1);
    end
    if sum(selector) > 0
        selector_ = [
            ((cross_pts(selector, 1) - x1(selector)) .* (cross_pts(selector, 1) .* x2(selector)) < 0) ...
            & ((cross_pts(selector, 2) - y1(selector)) .* (cross_pts(selector, 2) .* y2(selector)) < 0), ...
            ((cross_pts(selector, 3) - x1(selector)) .* (cross_pts(selector, 3) .* x2(selector)) < 0) ...
            & ((cross_pts(selector, 4) - y1(selector)) .* (cross_pts(selector, 4) .* y2(selector)) < 0), ...
            ((cross_pts(selector, 5) - x1(selector)) .* (cross_pts(selector, 5) .* x2(selector)) < 0) ...
            & ((cross_pts(selector, 6) - y1(selector)) .* (cross_pts(selector, 6) .* y2(selector)) < 0), ...
            ((cross_pts(selector, 7) - x1(selector)) .* (cross_pts(selector, 8) .* x2(selector)) < 0) ...
            & ((cross_pts(selector, 7) - y1(selector)) .* (cross_pts(selector, 8) .* y2(selector)) < 0), ...
            ];
        
        [~, yy] = find(selector_, 2);
    end
    %}
    %{
    for i = 1 : 12
        img = step(shapeInserter, img, int32([lines(i, 1) lines(i, 2) lines(i, 3) lines(i, 4)]));
    end
    %}
end