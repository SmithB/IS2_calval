
DEM_files=glob('/Volumes/ice2/ben/REMA_dems/16m/*.tif');
fid=fopen('ATM_reg_queue.txt','w');

for kf=1:length(DEM_files)
    fprintf(fid,'register_one_DEM(''%s'');\n', DEM_files{kf})
end
fclose(fid);

fields={'latitude','longitude','elevation'};

clear D2
dirs=glob('-d 20181114/20181114_ATM6dT7_rev01');
directory=dirs{1};
files=glob([directory,'/*.h5']);
for kf=1:length(files)
    thefile=files{kf};
    clear D0;
    D0.x=ll2ps(h5read(thefile,'/footprint/latitude'), h5read(thefile,'/footprint/longitude'));
    D0.h=h5read(thefile,'/footprint/elevation');
    D0=index_struct(D0, blockmedian(D0.x, D0.h, 20));
    D0.file_num=zeros(size(D0.h))+kf;
    D2(kf)=D0;
end

ff=fieldnames(D2);
for kf=1:length(ff)
    D.(ff{kf})=cat(1, D2.(ff{kf})); 
end
save ATM_blockmedian_data_longline D

clear D_IS;
mat_files=glob('/Volumes/ice2/ben/REMA_dems/16m/*.mat');
count=1;
for k=1:length(mat_files)
    L=load(mat_files{k});
    if isfield(L, 'IS'); 
        L.IS.file=mat_files{k};
        if all(size(L.IS.dx)==[2,3])
            D_IS(count)=L.IS; count=count+1;
        end
    end
end

clear IS
for field={'R','dx','dy','bias','model_slope','N'}
    IS.(field{1})=NaN(length(D_IS), 2, 3);
    for kP=1:3
        for kB=1:2
            for k=1:length(D_IS)
                if D_IS(k).N(kB, kP)>1000 && D_IS(k).R(kB, kP)<0.3
                    IS.(field{1})(k, kB, kP)=D_IS(k).(field{1})(kB, kP);
                end
            end
        end
    end
end





if false
    % Example presented at AGU 2018(?)
    II=read_geotif('/Volumes/ice2/ben/REMA_dems/16m/SETSM_WV01_20131213_1020010028D3B200_10200100271A0500_seg1_8m_v1.0_dem.tif')
    % mask the ATM data to the DEM
    els=real(D.x) > II.x(1)+20 & real(D.x) < II.x(end)-20 & imag(D.x) > II.y(1)+20 & imag(D.x) < II.y(end)-20;
    Dsub=index_struct(D, els);
    Dsub=index_struct(Dsub, isfinite(interp2(II.x, II.y, II.z(:,:,1), real(Dsub.x), imag(Dsub.x))));
    
    rows=II.y > min(imag(Dsub.x))-200 & II.y < max(imag(Dsub.x))+200;
    cols=II.x > min(real(Dsub.x))-200 & II.x < max(real(Dsub.x))+200;
    IIs=struct('x', II.x(cols),'y', II.y(rows), 'z', II.z(rows, cols, 1));
    ii=ceil((Dsub.x-(IIs.x(1)+1i*IIs.y(1)))/diff(IIs.x(1:2)));
    ii=unique(ii);
    ii=ii(real(ii)>0 & real(ii) < size(IIs.z,2) & imag(ii) > 0 & imag(ii) < size(IIs.z,1));
    rows=range(imag(ii)); cols=range(real(ii));
 
    mask=zeros(size(IIs.z));
    mask(sub2ind(size(mask), imag(ii), real(ii)))=1;
    K=exp(-(((-400:400)/100).^2));
    mask=conv2_separable(mask, K)>5 & isfinite(IIs.z);
    [gx, gy]=gradient(IIs.z, IIs.x, IIs.y);
    
    this_cmap=ones(60,3);
    this_cmap(1:30,1:2)=repmat(linspace(0, 1, 30)', [1,2]);
    this_cmap(31:end, 2:3)=repmat(linspace(1, 0, 30)', [1,2]);
    
    II_aligned=struct('x', II.x+ATM.dx,'y', II.y+ATM.dy, ...
        'z', II.z(:,:,1)-(ATM.m(1) + (II.x(:)'-real(ATM.x0))*ATM.m(2)+(II.y(:)-imag(ATM.x0))*ATM.m(3)));
    Dsub.DEM0=interp2(II.x, II.y, II.z(:,:,1), real(Dsub.x), imag(Dsub.x));
    Dsub.DEMc=interp2(II_aligned.x, II_aligned.y, II_aligned.z, real(Dsub.x), imag(Dsub.x));
    
    gxc=repmat(scale_to_byte(gx, [-1 1]*.02), [1 1 3]);
    dz_levels=[-2.9:.05:2.9];
    
    figure(7); clf;    set(gcf,'units','inches','position', [ 1.3733    3.4000    8.9467    5.4667],'color','w');

    hax=cheek_by_jowl(1, 4, [0 0.15 1 0.85]);
    
    axes(hax(1));
    imagesc(IIs.x+ATM.dx, IIs.y+ATM.dy, gxc,'alphadata', double(mask));  
    axis xy equal tight
    hold on;
    levels=linspace(min(Dsub.h), max(Dsub.h), 30)
    h_ATM=plot_colored_points(Dsub.x, Dsub.h, levels, [],flipud(jet(30))*.7+.3,  true);
    
    for kP=1:3
        els=isfinite(interp2(IIs.x, IIs.y, IIs.z(:,:,1), real(D6(kP).x), imag(D6(kP).x)));
        plot_colored_points(D6(kP).x(els), D6(kP).h_li(els), levels,  [],  flipud(jet), true);
    end
    h_bar(1)=axes('position', get(gca,'position').*[1 0 1 0]+[0 0.125 0 0.025],'fontsize', 15);
    imagesc(levels, [0 1], levels(:)'); 
    set(h_bar(1),'ytick', []); 
    colormap(h_bar(1),flipud(jet(30))*.7+.3)
    xlabel(h_bar(1),'WGS84 elevation, m');
    
    axes(hax(2));
    imagesc(IIs.x+ATM.dx, IIs.y+ATM.dy, gxc,'alphadata', double(mask));  hold on; axis xy equal tight
    plot_colored_points(Dsub.x, Dsub.h-Dsub.DEMc-nanmean(Dsub.h-Dsub.DEMc), dz_levels, [], this_cmap, true); 
    
     h_bar(2)=axes('position', get(gca,'position').*[1 0 1 0]+[0 0.125 0 0.025],'fontsize', 15);
    imagesc(dz_levels, [0 1], dz_levels(:)'); 
    set(h_bar(2),'ytick', []); 
    colormap(h_bar(2),this_cmap)
    xlabel(h_bar(2),'ATM - DEM, m');
  
    axes(hax(3));
    imagesc(IIs.x+ATM.dx, IIs.y+ATM.dy, gxc,'alphadata', double(mask)); hold on; axis xy equal tight
    for kP=1:3
        for kB=1:2
            plot_colored_points(IS.D(kB, kP).x, IS.D(kB, kP).z-IS.D(kB, kP).z_DEM, dz_levels, [], this_cmap, true)
        end
    end
    
    h_bar(3)=axes('position', get(gca,'position').*[1 0 1 0]+[0 0.125 0 0.025],'fontsize', 15);
    imagesc(dz_levels, [0 1], dz_levels(:)'); 
    set(h_bar(3),'ytick', []);
    colormap(h_bar(3),this_cmap)    
    xlabel(h_bar(3),'IS2_{raw} - DEM, m');

    axes(hax(4));
    imagesc(IIs.x+ATM.dx, IIs.y+ATM.dy, gxc,'alphadata', double(mask));  hold on; axis xy equal tight
    for kP=1:3
        for kB=1:2
            plot_colored_points(IS.D(kB, kP).x, IS.D(kB, kP).z-IS.D(kB, kP).zc, -3:.1:3, [], this_cmap, true)
        end
    end
     
    h_bar(4)=axes('position', get(gca,'position').*[1 0 1 0]+[0 0.125 0 0.025],'fontsize', 15);
    imagesc(dz_levels, [0 1], dz_levels(:)'); 
    set(h_bar(4),'ytick', []); 
    colormap(h_bar(4),this_cmap)
    xlabel(h_bar(4),'IS2_{corr} - DEM, m');
    set(hax,'ylim', range(imag(Dsub.x)),'xlim', range(real(Dsub.x)))
    set(hax,'visible','off')
    
    set(findobj(gcf,'type','line'),'markersize', 1)

end


    

