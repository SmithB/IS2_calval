function [Dp, channel_biases]=read_TEP_photons(thefile, DOPLOT, outfile)

if ~exist('DOPLOT','var')
    DOPLOT=true;
end

fields={'delta_time','ph_id_channel','ph_id_count','pce_mframe_cnt','tof_tep'};

hist_dt=1.25e-10;
hist_bins=0:hist_dt:1e-7;
bin_x=hist_bins(1:end-1)+0.5*diff(hist_bins(1:2));

channel_biases=NaN(120,1);

Dp=struct();
for kB=1:2
    for kF=1:length(fields)
        Dp(kB).(fields{kF})=h5read(thefile, sprintf('/pce%d/tep_photons/%s', kB, fields{kF}));
    end
    
    uC=unique(Dp(kB).ph_id_channel);
    nC=length(uC);%/2;
    
    [nn]=histcounts(Dp(kB).tof_tep, hist_bins);
    
    BG_region=[60 90]*1e-9;
    BG_els=bin_x>BG_region(1) & bin_x < BG_region(2);
    pulse_ctr=20e-9;
    pulse_W=6e-9;
    
    if DOPLOT
        figure; hold on;
    end
    
    offset_val=NaN(size(Dp(kB).ph_id_channel));
    for kC=1:nC
        els=Dp(kB).ph_id_channel==uC(kC); %| Dp(kB).ph_id_channel==uC(kC)+60;
        nn=histcounts(Dp(kB).tof_tep(els), hist_bins);
        BG_est(kC)=mean(nn(BG_els));
        this_ctr=pulse_ctr;
        for count=1:5
            % recenter the window on the pulse, recalculate the centroid
            pulse_els=bin_x>this_ctr-pulse_W/2 & bin_x < this_ctr+pulse_W/2;
            this_ctr=sum(bin_x(pulse_els).*(nn(pulse_els)-BG_est(kC)))/sum(nn(pulse_els)-BG_est(kC));
        end
        t_offset(kC)=this_ctr;
        offset_val(els)=t_offset(kC);
        channel_biases(uC(kC))=t_offset(kC);
        if DOPLOT
            plot(bin_x-this_ctr, histcounts(Dp(kB).tof_tep(els), hist_bins));
        end
    end
    
    if DOPLOT
        nn=histcounts(Dp(kB).tof_tep, hist_bins);
        plot(bin_x-mean(t_offset), nn/nC,'k','linewidth', 2);
        nn=histcounts(Dp(kB).tof_tep-offset_val, hist_bins-mean(t_offset));
        plot(bin_x-mean(t_offset), nn/nC,'r','linewidth', 3);
    end
    N_tot(kB)=length(Dp(kB).tof_tep);
end

if exist('outfile','var')
    save(outfile,'channel_biases', 'N_tot');
end

%set(gca,'xlim', [1.4 2.2]*1e-8);

% make a queue:
if false
    fid_in=fopen('list_of_good_Anc41s','r');
    fid_out=fopen('ANC41_queue.txt','w');
    while ~feof(fid_in)
        line=fgetl(fid_in);
        out_file=sprintf('channel_offsets/%s.mat', strrep(line,'.h5',''));
        fprintf(fid_out,'read_TEP_photons(''%s'', false,''%s'');\n', line, out_file);
    end
    fclose(fid_in);
    fclose(fid_out);
end

