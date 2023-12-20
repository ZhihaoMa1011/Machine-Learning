%% Read data
str=strcat('C:\Users\Zhihao\OneDrive - University of Utah\Redisual model\CSC\Matlab Figures and Codes\RD_EP_RC_5m_con.xlsx'); 
data=readtable(str);

t = 60;
tr = 60/t;

RDC = data{:,1};
RDH = data{:,2};

EPC = data{:,3};
EPH = data{:,4};

RC_32C = data{:,5};
RC_32H = data{:,6};

RC_21C = data{:,7};
RC_21H = data{:,8};

RDC(isnan(RDC)) = [];
RDH(isnan(RDH)) = [];
EPC(isnan(EPC)) = [];
EPH(isnan(EPH)) = [];
RC_32C(isnan(RC_32C)) = [];
RC_32H(isnan(RC_32H)) = [];
RC_21C(isnan(RC_21C)) = [];
RC_21H(isnan(RC_21H)) = [];

%% Order finding 
Gap=RDH-RC_32H;

p= parcorr(Gap,Numlags=300);

bounds_up = 0 + 2/(sqrt(length(Gap)));

bounds_low = 0 - 2/(sqrt(length(Gap)));


index = find(p>bounds_up | p<bounds_low); 
pacf = zeros(length(index),1);


for i=1:1:length(index)

    pacf(i) = p(index(i));

end

[B_abs,I] = sort(abs(pacf),'descend');

B = [pacf(I) index(I)];




Para = ones(11,4);

BIC = ones(10,10);


for j=1:1:10
    
    for n = 1:1:10

        try

            Mdl = arima('ARLags', (B(2:1+j,2)-ones(length(B(2:1+j,2)),1))','MALags',(B(2:1+n,2)-ones(length(B(2:1+n,2)),1))','D',1);
        
            resMdl  = estimate(Mdl,Gap);
        
            results = summarize(resMdl);
            
            BIC(j,n) = results.BIC;

            n = n+1

            j
                
            

        catch
    
            

        end
           
    end
   
    j = j+1;


end

[row1,column1]=find(BIC==1);

for i = 1:1:length(row1)

    BIC(row1(i,1),column1(i,1)) = BIC(row1(i,1)-1,column1(i,1)-1);

end




%% Model cv and mae with the finding order
[row,column]=find(BIC==min(min(BIC)))

Mdl = arima('ARLags', (B(2:row+1,2)-ones(length(B(2:row+1,2)),1))','MALags',(B(2:1+column,2)-ones(length(B(2:1+column,2)),1))','D',1);

resMdl  = estimate(Mdl,Gap);

results = summarize(resMdl);



residuals = infer(resMdl,Gap);

Gap_fix = Gap - residuals;



% % MAE and CV-RMSE FOR EPC
% EPC_f = EPC+Gap_fix;
% EPC_f(EPC_f < 0) = 0;
% 
% 
% MAE=1/(length(RDC))*sum(abs(RDC-EPC));
% MAEf=1/(length(RDC))*sum(abs(RDC-EPC_f))
% P1=(MAE-MAEf)/MAE
% 
% 
% RDC_m=sum(RDC)/length(RDC);
% CVRMSE=sqrt(1/(length(RDC))*sum((RDC-EPC).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDC))*sum((RDC-EPC_f).^2))/RDC_m
% P2=(CVRMSE-CVRMSEf)/CVRMSE

% % MAE and CV-RMSE FOR EPH
% EPH_f = EPH+Gap_fix;
% EPH_f(EPH_f < 0) = 0;
% 
% MAE=1/(length(RDH))*sum(abs(RDH-EPH));
% MAEf=1/(length(RDH))*sum(abs(RDH-EPH_f))
% P1=(MAE-MAEf)/MAE
% 
% 
% RDH_m=sum(RDH)/length(RDH);
% CVRMSE=sqrt(1/(length(RDH))*sum((RDH-EPH).^2))/RDH_m;
% CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-EPH_f).^2))/RDH_m
% P2=(CVRMSE-CVRMSEf)/CVRMSE

% % MAE and CV-RMSE FOR RCC
% RC_32C_f = RC_32C+Gap_fix;
% RC_32C_f(RC_32C_f < 0) = 0;
% 
% 
% MAE=1/(length(RDC))*sum(abs(RDC-RC_32C));
% MAEf=1/(length(RDC))*sum(abs(RDC-RC_32C_f))
% P1=(MAE-MAEf)/MAE
% 
% 
% RDC_m=sum(RDC)/length(RDC);
% CVRMSE=sqrt(1/(length(RDC))*sum((RDC-RC_32C).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDC))*sum((RDC-RC_32C_f).^2))/RDC_m
% P2=(CVRMSE-CVRMSEf)/CVRMSE


% MAE and CV-RMSE FOR RCH
RC_32H_f = RC_32H+Gap_fix;
RC_32H_f(RC_32H_f < 0) = 0;


MAE=1/(length(RDH))*sum(abs(RDH-RC_32H));
MAEf=1/(length(RDH))*sum(abs(RDH-RC_32H_f))
P1=(MAE-MAEf)/MAE


RDH_m=sum(RDH)/length(RDH);
CVRMSE=sqrt(1/(length(RDH))*sum((RDH-RC_32H).^2))/RDH_m;
CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-RC_32H_f).^2))/RDH_m
P2=(CVRMSE-CVRMSEf)/CVRMSE



% % MAE and CV-RMSE FOR RCH
% RC_21H_f = RC_21H+Gap_fix;
% RC_21H_f(RC_21H_f < 0) = 0;
% 
% MAE=1/(length(RDH))*sum(abs(RDH-RC_21H));
% MAEf=1/(length(RDH))*sum(abs(RDH-RC_21H_f));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% RDH_m=sum(RDH)/length(RDH);
% CVRMSE=sqrt(1/(length(RDH))*sum((RDH-RC_21H).^2))/RDH_m;
% CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-RC_21H_f).^2))/RDH_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;

% % MAE and CV-RMSE FOR RCC
% RC_21C_f = RC_21C+Gap_fix;
% RC_21C_f(RC_21C_f < 0) = 0;
% 
% t = 5;
% tr = 60/t;
% 
% 
% RDCa = RDC(3624*tr:6552*tr,1);
% RC_21C_fa = RC_21C_f(3624*tr:6552*tr,1);
% RC_21Ca = RC_21C(3624*tr:6552*tr,1);
% 
% MAE=1/(length(RDCa))*sum(abs(RDCa-RC_21Ca));
% MAEf=1/(length(RDCa))*sum(abs(RDCa-RC_21C_fa));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% RDC_m=sum(RDCa)/length(RDCa);
% CVRMSE=sqrt(1/(length(RDCa))*sum((RDCa-RC_21Ca).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDCa))*sum((RDCa-RC_21C_fa).^2))/RDC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;

% % MAE and CV-RMSE FOR EPC-3R2CC
% RC_32C_f = RC_32C+Gap_fix;
% RC_32C_f(RC_32C_f < 0) = 0;
% 
% EPCa = EPC(3624*tr:6552*tr,1);
% RC_32C_fa = RC_32C_f(3624*tr:6552*tr,1);
% RC_32Ca = RC_32C(3624*tr:6552*tr,1);
% 
% MAE=1/(length(EPCa))*sum(abs(EPCa-RC_32Ca));
% MAEf=1/(length(EPCa))*sum(abs(EPCa-RC_32C_fa));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% EPC_m=sum(EPCa)/length(EPCa);
% CVRMSE=sqrt(1/(length(EPCa))*sum((EPCa-RC_32Ca).^2))/EPC_m;
% CVRMSEf=sqrt(1/(length(EPCa))*sum((EPCa-RC_32C_fa).^2))/EPC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;

% % MAE and CV-RMSE FOR EPH-3R2CH
% RC_32H_f = RC_32H+Gap_fix;
% RC_32H_f(RC_32H_f < 0) = 0;
% 
% MAE=1/(length(EPH))*sum(abs(EPH-RC_32H));
% MAEf=1/(length(EPH))*sum(abs(EPH-RC_32H_f));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% EPH_m=sum(EPH)/length(EPH);
% CVRMSE=sqrt(1/(length(EPH))*sum((EPH-RC_32H).^2))/EPH_m;
% CVRMSEf=sqrt(1/(length(EPH))*sum((EPH-RC_32H_f).^2))/EPH_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;


% % MAE and CV-RMSE FOR EPC-2R1CC
% RC_21C_f = RC_21C+Gap_fix;
% RC_21C_f(RC_21C_f < 0) = 0;
% 
% EPCa = EPC(3624*tr:6552*tr,1);
% RC_21C_fa = RC_21C_f(3624*tr:6552*tr,1);
% RC_21Ca = RC_21C(3624*tr:6552*tr,1);
% 
% MAE=1/(length(EPCa))*sum(abs(EPCa-RC_21Ca));
% MAEf=1/(length(EPCa))*sum(abs(EPCa-RC_21C_fa));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% EPC_m=sum(EPCa)/length(EPCa);
% CVRMSE=sqrt(1/(length(EPCa))*sum((EPCa-RC_21Ca).^2))/EPC_m;
% CVRMSEf=sqrt(1/(length(EPCa))*sum((EPCa-RC_21C_fa).^2))/EPC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;

% % MAE and CV-RMSE FOR EPH-2R1CH
% RC_21H_f = RC_21H+Gap_fix;
% RC_21H_f(RC_21H_f < 0) = 0;
% 
% MAE=1/(length(EPH))*sum(abs(EPH-RC_21H));
% MAEf=1/(length(EPH))*sum(abs(EPH-RC_21H_f));
% P1=(MAE-MAEf)/MAE;
% Para(j,1) = MAEf;
% Para(j,2) = P1;
% 
% EPH_m=sum(EPH)/length(EPH);
% CVRMSE=sqrt(1/(length(EPH))*sum((EPH-RC_21H).^2))/EPH_m;
% CVRMSEf=sqrt(1/(length(EPH))*sum((EPH-RC_21H_f).^2))/EPH_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;
% Para(j,3) = CVRMSEf;
% Para(j,4) = P2;




%% plot BIC
figureUnits = 'centimeters';
figureWidth = 30;
figureHeight = 20;

figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);


h = heatmap(BIC);

colormap(navyblue);

     
         

h.Title = 'BIC for ARIMA (3R2C model heating energy 5 min)';
h.XLabel = 'MA';
h.YLabel = 'AR';
h.FontName = 'Calibri';
h.FontSize = 24;




figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
newFilePath = 'C:\Users\Zhihao\OneDrive - University of Utah\Redisual model\CSC\Matlab Figures and Codes\AR_order\';
fileout = [newFilePath 'heatmap_ARIMA_3R2C_H_5'];
print(figureHandle,[fileout,'.png'],'-r300','-dpng'); 
