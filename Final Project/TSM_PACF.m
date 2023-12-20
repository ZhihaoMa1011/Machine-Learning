% Read data
str=strcat('C:\Users\Zhihao\OneDrive - University of Utah\Redisual model\CSC\Matlab Figures and Codes\RD_EP_RC_60m_con.xlsx'); 
data=readtable(str);

RDC = data{:,1};
RDH = data{:,2};

EPC = data{:,3};
EPH = data{:,4};

RC_32C = data{:,5};
RC_32H = data{:,6};

RC_21C = data{:,7};
RC_21H = data{:,8};


Gap=RDH-RC_32H;
% econometricModeler;

CEPH = addcolor(213); 
CEPC = addcolor(222); 
CRDH = addcolor(215); 
CRDC = addcolor(212);
CRC32C = addcolor(231);
CRC32H = addcolor(211);
CRC21C = addcolor(144);
CRC21H = addcolor(30);


figureUnits = 'centimeters';
figureWidth = 70;
figureHeight = 20;


figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on;


p= parcorr(Gap,Numlags=300);

stem(0:1:300,p,'LineWidth',2,'Color',CRDH);


hold on; 

bounds_up = 0 + 2/(sqrt(length(Gap)));

bounds_low = 0 - 2/(sqrt(length(Gap)));


plot(bounds_up*ones(301), 'LineWidth',2,'Color','k');

hold on;

plot(bounds_low*ones(301), 'LineWidth',2,'Color','k');



set(gca, 'YColor', 'k',...
         'Box', 'on', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                        
         'TickDir', 'in', ...            
         'XMinorTick', 'off', 'YMinorTick', 'off', ...           
         'Xlim' , [0 300]) 

hYLabel = ylabel('Partial autocorrelation function');
hXLabel = xlabel('Lag (60 min)');

hLegend = legend(["Heating energy gap between observed data and 3R2C model" "Confident bounds (95%)"]);

set(gca, 'FontName', 'Calibri', 'FontSize', 36)
set([hYLabel,hXLabel], 'FontName',  'Calibri')
set([hYLabel,hXLabel], 'FontSize', 36)


figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'RD_3R2C_H_60m_PACF';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');


index = find(p>bounds_up | p<bounds_low); 
pacf = zeros(length(index),1);


for i=1:1:length(index)

    pacf(i) = p(index(i));

end

[B_abs,I] = sort(abs(pacf),'descend');

B = [pacf(I) index(I)];

%%

B1 = abs(B(1:20,1));
B2 = string(B(1:20,2)-1);

figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on;



stem(0:1:length(B1)-1,B1,'LineWidth',2,'Color',CRDH);


set(gca, 'YColor', 'k',...
         'Box', 'on', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                        
         'TickDir', 'in', ...            
         'XMinorTick', 'off', 'YMinorTick', 'off', ...           
         'Xlim' , [0 length(B1)]) 

xticklabels(B2);

set(gca,'xtick', 0:1:19);

hYLabel = ylabel('Partial autocorrelation function');
hXLabel = xlabel('Lag (60 min)');


set(gca, 'FontName', 'Calibri', 'FontSize', 36)
set([hYLabel,hXLabel], 'FontName',  'Calibri')
set([hYLabel,hXLabel], 'FontSize', 36)


figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'RD_3R2C_H_60m_PACF_order';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');




