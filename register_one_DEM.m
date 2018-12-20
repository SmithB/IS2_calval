function register_one_DEM(DEM_file)

[ATM_offsets.x, ATM_offsets.y]=meshgrid([-100:5:100]);
[IS2_offsets.x, IS2_offsets.y]=meshgrid([-300:10:300]);

load ATM_blockmedian_data_longline
out_file=strrep(DEM_file,'.tif','_ATM_reg.mat');

try
    II=read_geotif(DEM_file);
catch
    return
end
if exist(out_file,'file')
    load(out_file);
else
   
    els=real(D.x) > II.x(1)+20 & real(D.x) < II.x(end)-20 & imag(D.x) > II.y(1)+20 & imag(D.x) < II.y(end)-20;
    if sum(els) < 100
       return
    end
    Dsub=index_struct(D, els);
    Dsub=index_struct(Dsub, isfinite(interp2(II.x, II.y, II.z(:,:,1), real(Dsub.x), imag(Dsub.x))));
    if numel(Dsub.x) < 100; return; end
    
    R=zeros(size(ATM_offsets.x));
    G=[ones(size(Dsub.x(:))), real(Dsub.x(:)-mean(Dsub.x(:))), imag(Dsub.x-mean(Dsub.x))];
    
    for k=1:numel(R)
        Zi=interp2(II.x+ATM_offsets.x(k), II.y+ATM_offsets.y(k), II.z(:,:,1), real(Dsub.x), imag(Dsub.x));
        good=isfinite(Zi);
        m=G(good,:)\double(Zi(good)-Dsub.h(good));
        R(k)=std(Zi(good)-Dsub.h(good)-G(good,:)*m);
    end
    
    [~, best]=min(R(:));
    k=best;
    Zi=interp2(II.x+ATM_offsets.x(k), II.y+ATM_offsets.y(k), II.z(:,:,1), real(Dsub.x), imag(Dsub.x));
    good=isfinite(Zi);
    ATM.m=G(good,:)\double(Zi(good)-Dsub.h(good));
    ATM.R=R(best);
    ATM.dx=ATM_offsets.x(best);
    ATM.dy=ATM_offsets.y(best);
    ATM.N=sum(good);
    ATM.x0=mean(Dsub.x(:));
    fprintf(1,'%s dx=%3.2d, dy=%3.2d, R=%3.2d, N=%d\n', DEM_file, ATM.dx, ATM.dy, ATM.R, ATM.N);
    save(out_file, 'ATM')
end

% read in the IS2 data
D6=read_ASAS_ATL06('ATL06_v44/ATL06_20181114161346_07180111_944_01.h5');
for kp=1:3
    D6(kp).x=ll2ps(D6(kp).latitude, D6(kp).longitude);
    cloud_rat=D6(kp).n_fit_photons./D6(kp).w_surface_window_final;
    good_cloud=[cloud_rat(:,1)>1, cloud_rat(:,2) >4];
    good_AT=ATL06_AT_filter(D6(kp), 2);
    qs=D6(kp).h_robust_sprd>1 | D6(kp).h_li_sigma > 1  | D6(kp).snr_significance > 0.02;
    D6(kp).h_li(qs>0 | good_cloud==0 | good_AT==0)=NaN;
end

II_aligned=struct('x', II.x+real(ATM.dx),'y', II.y+imag(ATM.dx), ...
    'z', II.z(:,:,1)-(ATM.m(1) + (II.x(:)'-real(ATM.x0))*ATM.m(2)+(II.y(:)-imag(ATM.x0))*ATM.m(3)));


clear IS
for kP=1:3
    for kB=1:2        
        z0=interp2(II_aligned.x, II_aligned.y, II_aligned.z(:,:,1), real(D6(kP).x(:, kB)), imag(D6(kP).x(:, kB)));
        for field={'x','x_atc'}
            Dsub2.(field{1})=D6(kP).(field{1})(isfinite(z0), kB);
        end
        Dsub2.z=D6(kP).h_li(isfinite(z0), kB);
        
        G_AT=[ones(size(Dsub2.z(:))), Dsub2.x_atc(:)-mean(Dsub2.x_atc(:))];
        [R0_IS, B, S, N]=deal(NaN(size(IS2_offsets.x)));
        
        rr=1:5:size(IS2_offsets.x,1);
        cc=1:5:size(IS2_offsets.y,2);
        last_step=5;
        for step=[2 1 1]
            for kr=rr(:)'
                for kc=cc(:)'
                    k=sub2ind(size(IS2_offsets.x), kr, kc);
                    if isfinite(R0_IS(k)); continue; end
                    [R0_IS(k), m, N(k)] = shifted_misfit(II_aligned, Dsub2, IS2_offsets.x(k), IS2_offsets.y(k), G_AT);
                    B(k)=m(1); S(k)=m(2);
                end
            end
            [~, best]=min(R0_IS(:));
            [r,c]=ind2sub(size(R0_IS), best);
            cc=max(1, c-2*last_step):step:min(size(R0_IS,2), c+2*last_step);
            rr=max(1, r-2*last_step):step:min(size(R0_IS,1), r+2*last_step);
            last_step=step;
        end
        
        if ~any(isfinite(R0_IS(:)))
            continue
        end
        [IS.R(kB, kP), best]=min(R0_IS(:));
        IS.dx(kB, kP)=IS2_offsets.x(best);
        IS.dy(kB, kP)=IS2_offsets.y(best);
        [rr,cc]=find(R0_IS < R0_IS(best)*(1+1/sqrt(N(best))));
        IS.sigma_dx(kB, kP)=diff(range(IS2_offsets.x(1,cc)))/2;
        IS.sigma_dy(kB, kP)=diff(range(IS2_offsets.y(rr,1)))/2;
        IS.bias(kB, kP)=B(best);
        IS.model_slope(kB, kP)=S(best);
        IS.model_ctr(kB, kP)=mean(Dsub2.x_atc(:));
        IS.N(kB, kP)=N(best);        
        [~, ~, ~, z_est]=shifted_misfit(II_aligned, Dsub2, IS2_offsets.x(best), IS2_offsets.y(best), G_AT);  
        Dsub2.zc=z_est;
        Dsub2.z_DEM=interp2(II_aligned.x, II_aligned.y, II_aligned.z, real(Dsub2.x), imag(Dsub2.x));
        IS.D(kB, kP)=Dsub2;
    end
end
if exist('IS','var')
    save(out_file, 'ATM', 'IS')
end

if false
    % mask the ATM data to the DEM
    els=real(D.x) > II.x(1)+20 & real(D.x) < II.x(end)-20 & imag(D.x) > II.y(1)+20 & imag(D.x) < II.y(end)-20;
    Dsub=index_struct(D, els);
    Dsub=index_struct(Dsub, isfinite(interp2(II.x, II.y, II.z(:,:,1), real(Dsub.x), imag(Dsub.x))));
    
    for kP=1:3
        els=isfinite(interp2(II.x, II.y, II.z(:,:,1), real(D6(kP).x), imag(D6(kP).x)));
        plot_colored_points(D6(kP).x(els), D6(kP).h_li(els), [1700:5:1780], [], flipud(jet),true);
    end
end

%--------------------------------------------------------------------------
function [R0, m, N, d_est] = shifted_misfit(II, D, dx, dy, G)

[R0, B, S, N]=deal(NaN);
z1=interp2(II.x+dx, II.y+dy, II.z, real(D.x), imag(D.x), '*linear');
good=isfinite(z1);
if sum(good) < 100
    return
end
m=G(good,:)\double(D.z(good)-z1(good));
d_est=z1+G*m;
R0=std(D.z(good)-d_est(good));
N=sum(good);

