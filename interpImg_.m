function interVal = interpImg_(image , pts_2d)
    sz = size(image);
    try
        pts_2d1 = floor(pts_2d);
        pts_2d2 = ceil(pts_2d);
        linear_indlt = sub2ind(sz, pts_2d1(:,2), pts_2d1(:,1));
        linear_indrt = sub2ind(sz, pts_2d1(:,2), pts_2d2(:,1));
        linear_indlb = sub2ind(sz, pts_2d2(:,2), pts_2d1(:,1));
        linear_indrb = sub2ind(sz, pts_2d2(:,2), pts_2d2(:,1));
        val_lt = image(linear_indlt);
        val_rt = image(linear_indrt);
        val_lb = image(linear_indlb);
        val_rb = image(linear_indrb);
        
        val1 = (pts_2d2(:,1) - pts_2d(:,1)) .* val_lb + (1 - (pts_2d2(:,1) - pts_2d(:,1))) .* val_rb;
        val2 = (pts_2d2(:,1) - pts_2d(:,1)) .* val_lt + (1 - (pts_2d2(:,1) - pts_2d(:,1))) .* val_rt;
        interVal = (pts_2d2(:,2) - pts_2d(:,2)) .* val2 + (1 - (pts_2d2(:,2) - pts_2d(:,2))) .* val1;
    catch
        pts_2d(pts_2d(:,1) < 1, 1) = 1;
        pts_2d(pts_2d(:,1) > sz(2), 1) = sz(2);
        pts_2d(pts_2d(:,2) < 1, 2) = 1;
        pts_2d(pts_2d(:,2) > sz(1), 2) = sz(1);
        
        pts_2d1 = floor(pts_2d);
        pts_2d2 = ceil(pts_2d);
        linear_indlt = sub2ind(sz, pts_2d1(:,2), pts_2d1(:,1));
        linear_indrt = sub2ind(sz, pts_2d1(:,2), pts_2d2(:,1));
        linear_indlb = sub2ind(sz, pts_2d2(:,2), pts_2d1(:,1));
        linear_indrb = sub2ind(sz, pts_2d2(:,2), pts_2d2(:,1));
        val_lt = image(linear_indlt);
        val_rt = image(linear_indrt);
        val_lb = image(linear_indlb);
        val_rb = image(linear_indrb);
        
        val1 = (pts_2d2(:,1) - pts_2d(:,1)) .* val_lb + (1 - (pts_2d2(:,1) - pts_2d(:,1))) .* val_rb;
        val2 = (pts_2d2(:,1) - pts_2d(:,1)) .* val_lt + (1 - (pts_2d2(:,1) - pts_2d(:,1))) .* val_rt;
        interVal = (pts_2d2(:,2) - pts_2d(:,2)) .* val2 + (1 - (pts_2d2(:,2) - pts_2d(:,2))) .* val1;
    end
end
