clear chb;
files=glob('ANC41/channel_offsets/ANC41*.mat');
files=files(1:end-1);
for k=1:length(files)
    load(files{k});
    chb(:,k)=channel_biases(:);
end

clf;hold on
set(gca,'colororder',lines(60));
plot(chb(1:60,:)*1e9);
plot(chb(61:end,:)*1e9);
xlabel('pixel'); ylabel('offset, ns');

files=glob('AA_03/deadtime_differences/201/*.txt');
for kf=1:length(files)
    CB0{kf}=load(files{kf});
end
CB=cat(1, CB0{:});

 % (channel, delta_t.size, Dch['h_ph'].size/np.float(D['h_ph'].size)*n_px, N_plus, N_minus, dt_est_minus[channel]*1.e9, dt_est_plus[channel]*1.e9))

% channel, N, f, N_plus, N_minus, dt_minus, dt_plus;
uC=unique(CB(:,1));
figure(2); clf; h=cheek_by_jowl(6, 10, [0 0 1 1]);
for k=1:length(uC)
    els=find(CB(:,1)==uC(k));
    delta=CB(els,end)-CB(els, end-1);
    axes(h(k));
    histogram(delta, -1:.1:1);
    mDelta(k)=median(delta);
end

mchb=mean(chb,2);
mchb_corr=mchb;
mchb_corr(1:60)=mchb(1:60)*1e9-mDelta'/4;
mchb_corr(61:end)=mchb(61:end)*1e9+mDelta'/4;

figure(3); hold on; clf; 
plot(mchb*1e9,'r');
hold on;
plot(mchb_corr,'k');
xlabel('pixel');
ylabel('bias, ns');
legend('raw','corrected');





