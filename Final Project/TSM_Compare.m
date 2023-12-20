% Read data
str=strcat('C:\Users\Zhihao\OneDrive - University of Utah\Redisual model\CSC\Matlab Figures and Codes\RD_EP_RC_5m_con.xlsx'); 
data=readtable(str);

RDC = data{:,1};
RDH = data{:,2};

EPC = data{:,3};
EPH = data{:,4};

RC_32C = data{:,5};
RC_32H = data{:,6};

RC_21C = data{:,7};
RC_21H = data{:,8};

L = length(RDH);


CEPH = addcolor(213); 
CEPC = addcolor(222); 
CRDH = addcolor(215); 
CRDC = addcolor(212);
CRC32C = addcolor(231);
CRC32H = addcolor(211);
CRC21C = addcolor(144);
CRC21H = addcolor(30);

figureUnits = 'centimeters';
figureWidth = 30;
figureHeight = 20;


figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on;

f1 = plot(RDH,'color',CRDH,'linewidth',1);

hold on;

f2 = plot(RC_32H,'color',CRC32H,'linewidth',1);

set(gca, 'YColor', 'k',...
         'Box', 'on', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                        
         'TickDir', 'in', ...            
         'XMinorTick', 'off', 'YMinorTick', 'off', ...           
         'YTick', 0:25:750,...
         'XTick', [],...  
         'Xlim' , [0 L],...
         'Ylim' , [0 75]) 

hYLabel = ylabel('Heating energy (kwh)');
hXLabel = xlabel('Time (5 min)');

hLegend = legend([f1, f2], ...
          'Observed data', '3R2C model', ...
                 'Location', 'northwest');


set(gca, 'FontName', 'Calibri', 'FontSize', 36)
set([hYLabel,hXLabel], 'FontName',  'Calibri')
set([hYLabel,hXLabel], 'FontSize', 36)


figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'RD_RC32_H_5m';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');


%%
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on;


f3 = plot(RDH-RC_32H,'color',CRC32H,'linewidth',1);

set(gca, 'YColor', 'k',...
         'Box', 'on', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                        
         'TickDir', 'in', ...            
         'XMinorTick', 'off', 'YMinorTick', 'off', ...           
         'YTick', -25:25:75,...
         'XTick', [],...  
         'Xlim' , [0 L],...
         'Ylim' , [-25 75]) 

hYLabel = ylabel('Heating energy gap (kwh)');
hXLabel = xlabel('Time (5 min)');



set(gca, 'FontName', 'Calibri', 'FontSize', 36)
set([hYLabel,hXLabel], 'FontName',  'Calibri')
set([hYLabel,hXLabel], 'FontSize', 36)


figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'RD_RC32_H_5m_gap';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');
