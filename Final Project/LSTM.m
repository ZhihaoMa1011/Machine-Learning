clear;

% Read data
str=strcat('C:\Users\Zhihao\OneDrive - University of Utah\Redisual model\CSC\Matlab Figures and Codes\RD_EP_RC_60m_con.xlsx'); 
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

%% order finding

Gap=RDC-EPC;


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


%% LSTM

B = B

M_i = max(B);

L = length(B);

P_train = [];

for i = 1:1:L

    P_train = [P_train Gap(M_i+1-i:length(Gap)-i,:)];

end
P_train = P_train';
T_train = Gap(M_i+1:length(Gap),:)';
M = size(P_train,2);



[P_train, ps_input] = mapminmax(P_train,0,1);


[t_train, ps_output] = mapminmax(T_train,0,1);


P_train = double(reshape(P_train ,L,1,1,M));


t_train = t_train';



for i = 1:M
    p_train{i,1} = P_train(:,:,1,i);
end



layers = [
    sequenceInputLayer(L)

    lstmLayer(4, 'OutputMode','last')
    reluLayer

    fullyConnectedLayer(1)
    regressionLayer];

options = trainingOptions('adam', ...
    'MiniBatchSize',20, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',500, ...
    'Plots','training-progress', ...
    'Verbose',false);



options = trainingOptions('sgdm', 'OutputFcn', @customOutputFunction);

net = trainNetwork(p_train, t_train, layers, options);

t_sim1 = predict(net,p_train);
lossValues = evalin('base', 'storedLossValues');

T_sim1 = mapminmax('reverse', t_sim1, ps_output);




T_sim1 = [T_sim1(1:M_i,1); T_sim1];


para = ones(1,4);


% % MAE and CV-RMSE FOR RDC-EPC
% EPC_f = EPC+T_sim1;
% EPC_f(EPC_f < 0) = 0;
% 
% % 60m 30m 15m 5m, t is the time resolution, xxxa is the data from Jun
% % to Sep
% 
% % t = 5;
% % tr = 60/t;
% 
% 
% RDCa = RDC(3624*tr:6552*tr,1);
% EPC_fa = EPC_f(3624*tr:6552*tr,1);
% EPCa = EPC(3624*tr:6552*tr,1);
% 
% 
% MAE=1/(length(RDCa))*sum(abs(RDCa-EPCa));
% MAEf=1/(length(RDCa))*sum(abs(RDCa-EPC_fa));
% P1=(MAE-MAEf)/MAE;
% 
% 
% RDC_m=sum(RDCa)/length(RDCa);
% CVRMSE=sqrt(1/(length(RDCa))*sum((RDCa-EPCa).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDCa))*sum((RDCa-EPC_fa).^2))/RDC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;


% 
% % MAE and CV-RMSE FOR RDH-EPH
% EPH_f = EPH+T_sim1;
% EPH_f(EPH_f < 0) = 0;
% 
% MAE=1/(length(RDH))*sum(abs(RDH-EPH));
% MAEf=1/(length(RDH))*sum(abs(RDH-EPH_f));
% P1=(MAE-MAEf)/MAE;
% 
% 
% RDH_m=sum(RDH)/length(RDH);
% CVRMSE=sqrt(1/(length(RDH))*sum((RDH-EPH).^2))/RDH_m;
% CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-EPH_f).^2))/RDH_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;






% % MAE and CV-RMSE FOR RDC-3R2C
% RC_32C_f = RC_32C+T_sim1;
% RC_32C_f(RC_32C_f < 0) = 0;
% 
% % t = 15;
% % tr = 60/t;
% 
% 
% RDCa = RDC(3624*tr:6552*tr,1);
% RC_32C_fa = RC_32C_f(3624*tr:6552*tr,1);
% RC_32Ca = RC_32C(3624*tr:6552*tr,1);
% 
% MAE=1/(length(RDCa))*sum(abs(RDCa-RC_32Ca));
% MAEf=1/(length(RDCa))*sum(abs(RDCa-RC_32C_fa));
% P1=(MAE-MAEf)/MAE;
% 
% 
% RDC_m=sum(RDCa)/length(RDCa);
% CVRMSE=sqrt(1/(length(RDCa))*sum((RDCa-RC_32Ca).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDCa))*sum((RDCa-RC_32C_fa).^2))/RDC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;


% MAE and CV-RMSE FOR RCH
RC_32H_f = RC_32H+T_sim1;
RC_32H_f(RC_32H_f < 0) = 0;

MAE=1/(length(RDH))*sum(abs(RDH-RC_32H));
MAEf=1/(length(RDH))*sum(abs(RDH-RC_32H_f));
P1=(MAE-MAEf)/MAE;


RDH_m=sum(RDH)/length(RDH);
CVRMSE=sqrt(1/(length(RDH))*sum((RDH-RC_32H).^2))/RDH_m;
CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-RC_32H_f).^2))/RDH_m;
P2=(CVRMSE-CVRMSEf)/CVRMSE;






% % MAE and CV-RMSE FOR RCC
% RC_21C_f = RC_21C+T_sim1;
% RC_21C_f(RC_21C_f < 0) = 0;
% 
% % t = 5;
% % tr = 60/t;
% 
% 
% RDCa = RDC(3624*tr:6552*tr,1);
% RC_21C_fa = RC_21C_f(3624*tr:6552*tr,1);
% RC_21Ca = RC_21C(3624*tr:6552*tr,1);
% 
% MAE=1/(length(RDCa))*sum(abs(RDCa-RC_21Ca));
% MAEf=1/(length(RDCa))*sum(abs(RDCa-RC_21C_fa));
% P1=(MAE-MAEf)/MAE;
% 
% RDC_m=sum(RDCa)/length(RDCa);
% CVRMSE=sqrt(1/(length(RDCa))*sum((RDCa-RC_21Ca).^2))/RDC_m;
% CVRMSEf=sqrt(1/(length(RDCa))*sum((RDCa-RC_21C_fa).^2))/RDC_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;



% % MAE and CV-RMSE FOR RCH
% RC_21H_f = RC_21H+T_sim1;
% RC_21H_f(RC_21H_f < 0) = 0;
% 
% MAE=1/(length(RDH))*sum(abs(RDH-RC_21H));
% MAEf=1/(length(RDH))*sum(abs(RDH-RC_21H_f));
% P1=(MAE-MAEf)/MAE;
% 
% RDH_m=sum(RDH)/length(RDH);
% CVRMSE=sqrt(1/(length(RDH))*sum((RDH-RC_21H).^2))/RDH_m;
% CVRMSEf=sqrt(1/(length(RDH))*sum((RDH-RC_21H_f).^2))/RDH_m;
% P2=(CVRMSE-CVRMSEf)/CVRMSE;




para(1,1) = MAEf;
para(1,2) = P1;
para(1,3) = CVRMSEf;
para(1,4) = P2;

