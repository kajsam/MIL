function AUC = classification_divergences(condI_pos, condI_neg, ratioI, bag_class, fig, color)

% Classification by simple threshold.
% Kajsa M?llersen (kajsa.mollersen@uit.no) 2017.01.27

% Input     :
%           condI_pos - Conditional KL information (on pos class)
%           condI_neg - Conditional KL information (on neg class)
%           ratioI    - Ratio of KL informations

n_bag = length(bag_class);

% Sort the bag-to-class divergence values in descending order
[sort_condI_pos, idx] = sort(condI_pos,'descend');
sbag_class = bag_class(idx);     % and the class labels
idx_pos = find(sbag_class == 2); % identify the positive bags
n_pos = length(idx_pos);

% Set initial values for sensitivity and specificity
SE = zeros(1,n_pos+2); SP = ones(1,n_pos+2);
SE(end) = 1; SP(end) = 0;
for j = 1: n_pos
  label = ones(1,n_bag);      % All bags are negative
  % Define threshold at pos bag with jth largest value
  thresh = sort_condI_pos(idx_pos(j)); 
  label(sort_condI_pos>=thresh) = 2;   % All bags above threshold are positive
  
  % Calculate sensitivity and specificity
  CP = classperf(sbag_class,label,'Positive', 2, 'Negative', 1);    
  SE(j+1) = CP.Sensitivity;
  SP(j+1) = CP.Specificity;
end
cond_pos_AUC = round(100*trapz(1-SP,SE))/100; % Area under ROC curve

if fig
  figure(fig), subplot(1,3,1), hold on, grid on, title('Bag-to-class')
  plot(1-SP,SE,'Color',color)
end 

% Repeat for conditioning on negative class
[sort_condI_neg, idx] = sort(condI_neg,'ascend'); % 'ascend'
sbag_class = bag_class(idx);     
idx_pos = find(sbag_class == 2); 

nSE = zeros(1,n_pos+2); nSP = ones(1,n_pos+2);
nSE(1) = 0; nSP(1) = 1; nSE(end) = 1; nSP(end) = 0;
for j = 1: n_pos
  label = ones(1,n_bag);     % 2* 
  thresh = sort_condI_neg(idx_pos(j)); 
  label(sort_condI_neg<thresh) = 2;   % All bags above threshold are negative
  
  % Calculate sensitivity and specificity
  CP = classperf(sbag_class,label,'Positive', 2, 'Negative', 1);    
  nSE(j+1) = CP.Sensitivity;
  nSP(j+1) = CP.Specificity;
end
cond_neg_AUC = round(100*trapz(1-nSP,nSE))/100; % Area under ROC curve
  
if fig
  figure(fig), subplot(1,3,2), hold on, grid on, title('Bag-to-class (neg)')
  plot(1-nSP,nSE,'Color',color)
end

% Repeat for divergence ratio
[sort_ratioI, idx] = sort(ratioI,'descend');
sbag_class = bag_class(idx);
idx_pos = find(sbag_class == 2);

rSE = zeros(1,n_pos+2); rSP = ones(1,n_pos+2);
rSE(end) = 1; rSP(end) = 0;
for j = 1: n_pos
  label = ones(1,n_bag);
  thresh = sort_ratioI(idx_pos(j));
  label(sort_ratioI>=thresh) = 2;
 
  CP = classperf(sbag_class,label,'Positive', 2, 'Negative', 1);    
  rSE(j+1) = CP.Sensitivity;
  rSP(j+1) = CP.Specificity;
end
ratio_AUC = round(100*trapz(1-rSP,rSE))/100;

if fig
  figure(fig), subplot(1,3,3), hold on, grid on, title('Ratio')
  plot(1-rSP,rSE,'Color',color)
end
  
AUC = [cond_pos_AUC cond_neg_AUC ratio_AUC];

save_to_base(1)