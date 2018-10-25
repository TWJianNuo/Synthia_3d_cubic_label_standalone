function cuboid = generate_cuboid_by_center_(xc, yc, theta, l, w, h)
    bt_pt1 = [
        xc - cos(theta) * l / 2 + sin(theta) * w / 2;
        yc - sin(theta) * l / 2 - cos(theta) * w / 2;
        0;
        ]';
    bt_pt2 = [
        xc + cos(theta) * l / 2 + sin(theta) * w / 2;
        yc + sin(theta) * l / 2 - cos(theta) * w / 2;
        0;
        ]';
    bt_pt3 = [
        xc + cos(theta) * l / 2 - sin(theta) * w / 2;
        yc + sin(theta) * l / 2 + cos(theta) * w / 2;
        0;
        ]';
    bt_pt4 = [
        xc - cos(theta) * l / 2 - sin(theta) * w / 2;
        yc - sin(theta) * l / 2 + cos(theta) * w / 2;
        0;
        ]';
    top_pt1 = bt_pt1 + [0 0 h];
    top_pt2 = bt_pt2 + [0 0 h];
    top_pt3 = bt_pt3 + [0 0 h];
    top_pt4 = bt_pt4 + [0 0 h];
    cuboid = cell([6 1]);
    cuboid{1}.theta = theta;
    cuboid{1}.pts = [bt_pt1; bt_pt2; top_pt2; top_pt1];
    cuboid{2}.pts = [bt_pt2; bt_pt3; top_pt3; top_pt2];
    cuboid{3}.pts = [bt_pt3; bt_pt4; top_pt4; top_pt3];
    cuboid{4}.pts = [bt_pt4; bt_pt1; top_pt1; top_pt4];
    cuboid{5}.pts = [top_pt1; top_pt2; top_pt3; top_pt4];
    cuboid{1}.params = [-sin(theta), cos(theta), 0, -yc * cos(theta) + xc * sin(theta) + w / 2];
    cuboid{2}.params = [-cos(theta), -sin(theta), 0, yc * sin(theta) + xc * cos(theta) + l / 2];
    cuboid{3}.params = [sin(theta), -cos(theta), 0, -xc * sin(theta) + yc * cos(theta) + w / 2];
    cuboid{4}.params = [cos(theta), sin(theta), 0, -xc * cos(theta) - yc * sin(theta) + l / 2];
    cuboid{1}.length1 = l; cuboid{1}.length2 = h;
    cuboid{2}.length1 = w; cuboid{2}.length2 = h;
    cuboid{3}.length1 = l; cuboid{3}.length2 = h;
    cuboid{4}.length1 = w; cuboid{4}.length2 = h;
    cuboid{5}.length1 = l; cuboid{5}.length2 = w;
    cuboid{1}.plane_dirs = [cos(theta), sin(theta), 0, ];
    cuboid{2}.plane_dirs = [-sin(theta), cos(theta), 0];
    cuboid{3}.plane_dirs = [-cos(theta), -sin(theta), 0];
    cuboid{4}.plane_dirs = [sin(theta), -cos(theta), 0];
end

function test()
    for i = 1 : 1000
        xc = (rand(1) - 1/2) * 2 * pi; yc = rand(1); theta = rand(1); l = rand(1); w = rand(1); h = rand(1);
        cuboid_ = generate_cuboid_by_center(xc, yc, theta, l, w, h);
        bt_pt1 = [
            xc - cos(theta) * l / 2 + sin(theta) * w / 2;
            yc - sin(theta) * l / 2 - cos(theta) * w / 2;
            0;
            ]';
        bt_pt2 = [
            xc + cos(theta) * l / 2 + sin(theta) * w / 2;
            yc + sin(theta) * l / 2 - cos(theta) * w / 2;
            0;
            ]';
        bt_pt3 = [
            xc + cos(theta) * l / 2 - sin(theta) * w / 2;
            yc + sin(theta) * l / 2 + cos(theta) * w / 2;
            0;
            ]';
        bt_pt4 = [
            xc - cos(theta) * l / 2 - sin(theta) * w / 2;
            yc - sin(theta) * l / 2 + cos(theta) * w / 2;
            0;
            ]';
        top_pt1 = bt_pt1 + [0 0 h];
        top_pt2 = bt_pt2 + [0 0 h];
        top_pt3 = bt_pt3 + [0 0 h];
        top_pt4 = bt_pt4 + [0 0 h];
        cuboid = cell([6 1]);
        cuboid{1}.theta = theta;
        cuboid{1}.pts = [bt_pt1; bt_pt2; top_pt2; top_pt1];
        cuboid{2}.pts = [bt_pt2; bt_pt3; top_pt3; top_pt2];
        cuboid{3}.pts = [bt_pt3; bt_pt4; top_pt4; top_pt3];
        cuboid{4}.pts = [bt_pt4; bt_pt1; top_pt1; top_pt4];
        cuboid{5}.pts = [top_pt1; top_pt2; top_pt3; top_pt4];
        cuboid{1}.params = [-sin(theta), cos(theta), 0, -yc * cos(theta) + xc * sin(theta) + w / 2];
        cuboid{2}.params = [-cos(theta), -sin(theta), 0, yc * sin(theta) + xc * cos(theta) + l / 2];
        cuboid{3}.params = [sin(theta), -cos(theta), 0, -xc * sin(theta) + yc * cos(theta) + w / 2];
        cuboid{4}.params = [cos(theta), sin(theta), 0, -xc * cos(theta) - yc * sin(theta) + l / 2];
        cuboid{1}.length1 = l; cuboid{1}.length2 = h;
        cuboid{2}.length1 = w; cuboid{2}.length2 = h;
        cuboid{3}.length1 = l; cuboid{3}.length2 = h;
        cuboid{4}.length1 = w; cuboid{4}.length2 = h;
        cuboid{5}.length1 = l; cuboid{5}.length2 = w;
        cuboid{1}.plane_dirs = [cos(theta), sin(theta), 0, ];
        cuboid{2}.plane_dirs = [-sin(theta), cos(theta), 0];
        cuboid{3}.plane_dirs = [-cos(theta), -sin(theta), 0];
        cuboid{4}.plane_dirs = [sin(theta), -cos(theta), 0];
        th = 1e-6;
        for j = 1 : 5
            if max(max(abs(cuboid{j}.pts - cuboid_{j}.pts))) > th
                disp('Error')
            end
            if max(max(abs(cuboid{j}.length1 - cuboid_{j}.length1))) > th
                disp('Error')
            end
            if max(max(abs(cuboid{j}.length2 - cuboid_{j}.length2))) > th
                disp('Error')
            end
        end
        for j = 1 : 4
            if max(max(abs(cuboid{j}.params - cuboid_{j}.params))) > th
                disp('Error')
            end
            if max(max(abs(cuboid{j}.plane_dirs - cuboid_{j}.plane_dirs(1,:)))) > th
                disp('Error')
            end
        end
    end
end