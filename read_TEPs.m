% ANC41 fields to read:
fields={'tep_bckgrd','tep_hist','tep_hist_sum','tep_hist_x','tep_index_beg','tep_ph_cnt'};

clear D
d=dir('ANC*.h5');
count=0;
good_file_list=cell(1);
for k=1:length(d)
    clear D0;
    try
        % try to read the histogram groups, skip the file if they're
        % missing
        for kF=1:length(fields)
            for kB=1:2
                D0(kB).(fields{kF})=double(h5read(d(k).name, sprintf('/pce%d/tep_histogram/%s', kB, fields{kF})));
            end
        end
    catch
        continue
    end
    if exist('D0','var')
        count=count+1;
        good_file_list{count}=d(k).name;
        for kB=1:2
            % add teh background back to the TEP histogram
            D0(kB).tep_hist=D0(kB).tep_hist+(D0(kB).tep_bckgrd./D0(kB).tep_ph_cnt)';
            D0(kB).file_number=count+zeros(size(D0(kB).tep_ph_cnt));           
            %Hmax=max(D0(kB).tep_hist(500:800,:))./max(D0(kB).tep_hist);
            %if any(Hmax<0.005)
                %figure(count); plot(D0(kB).tep_hist_x, D0(kB).tep_hist(:, Hmax<0.005)); hold on; set(gca,'yscale','log');
            %end
        end
        D(count,:)=D0;
    end
end
clear WF_est
figure;clf;  hold on;
colors={'r','b'};
for kB=1:2;
    % put together all the histograms
    H=cat(2, D(:,kB).tep_hist);
    %Hn=max(H(500:800,:))./max(H);
    Hn=std(H(500:800,:));
    % undo the normalizations
    C=H.*cat(1, D(:,kB).tep_ph_cnt)';
    % smooth out the binning artifacts
    tK=[-10:10]'; K=exp(-(tK/4).^2);
    M=conv2(C, K/sum(K),'same');
    Mbar=mean(M(:, Hn<2e-5), 2);
    
    % subtract the pre-leading-edge noise
    Mbar_corr=max(0,Mbar-mean(Mbar(204:276)));
    
    Mbar_norm=Mbar_corr/sum(Mbar_corr(D(1).tep_hist_x<3e-8));
    ii=D(1).tep_hist_x<4e-8;
    WF_est(kB).t=D(1).tep_hist_x(ii,1);
    WF_est(kB).p=Mbar_norm(ii);
    t0(kB)=sum(WF_est(kB).t.*WF_est(kB).p)./sum(WF_est(kB).p);
    WF_est(kB).t=WF_est(kB).t-t0(kB);
    
    
    hp(kB)=plot((D(kB).tep_hist_x-t0(kB))*1e9, Mbar_norm,'.','color', colors{kB});
    if false
        % code to plot hte probability of detectino in a 3 ns or w ns
        % window
        P_3ns=conv(Mbar_norm, ones(ceil(3.2e-9/diff(D(1).tep_hist_x(1:2))),1),'same');
        P_1ns=conv(Mbar_norm, ones(ceil(1e-9/diff(D(1).tep_hist_x(1:2))),1),'same');
        h3(kB)=plot((D(kB).tep_hist_x-t0(kB))*1e9, P_3ns,'--','color', colors{kB});
        h1(kB)=plot((D(kB).tep_hist_x-t0(kB))*1e9, P_1ns,'-','color', colors{kB});
    end
end
set(findobj(gca,'type','line'),'linewidth', 2);

xlabel(gca,'t-t_{tx}, ns');
if ~exist(h3,'var')
    ylabel('PDF ');
    legend([hp(1)], 'PDF');
else
    
    ylabel('PDF (dots), P/t_{dead} (dashes)');
    legend([hp(1), h3(1), h1(1)], {'PDF','P_{3 ns}','P_{1 ns}'});
end

WF=WF_est(1);
% write out the waveform estimate
WF.p(WF.t<-2.95e-9)=10*eps;
 
 
 
 
 
 