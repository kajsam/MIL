function breast_cancer_main

% Requires: gaussian_mixture.m, bag_to_class_divergence.m, classification_divergences
%           breast_mil_div.m


% Data from http://www.miproblems.org/datasets/ucsb-breast/

% "UCSB Breast is an image classification problem. The original datasets 
% consists of 58 TMA image excerpts (896 x 768 pixels) taken from 32 benign 
% and 26 malignant breast cancer patients. The learning task is to classify 
% images as benign (negative) or malignant (positive).

% Patches of 7x7 size are extracted. The image is thresholded to segment 
% the content from the white background and the patches that contain 
% background more than 75% of their area are discarded. The features used 
% are 657 features that are global to the patch (histogram, LBP, SIFT), and 
% averaged features extracted from the cells, detected in each patch."

% I have tried some different approaches in 'breast_cancer_fold.m', and
% this is what I have settled at

addpath('/Volumes/kam025/Documents/MATLAB/MIL')

% Load the data

rng('default') % For reproducibility
warning off all % This will tell you if MaxIter reaches its limit

load ucsb_breast.mat
data = x.data; % 2002x708 (instances x feature values)   [1:236 658:end]

% Normalise
% First, exclude features with zero variance
vr = zeros(1,size(data,2));
for i = 1: size(data,2)
  vr(i) = var(data(:,i));
end
data(:,vr ==0) = [];

for i = 1: size(data,2)
  data(:,i) = data(:,i)-mean(data(:,i));
  data(:,i) = data(:,i)./var(data(:,i));
end

% There are way too many features in this data set. Even if PCA is used,
% redundant and irrelevant features decreases performance. But there aren't
% enough data to do feature selection based on feature values. 
feat_red = 0;
if feat_red
  for i = 1: 114
    R = corr(data);
    [row,col] = find(abs(R)>0.9); 
    delidx = row((row ~= col));
    data(:,delidx(1)) = [];
  end

  R = corr(data);
  max(abs(R(R<1)))

  vr = zeros(1,size(data,2));
  for i = 1: size(data,2)
    vr(i) = var(data(:,i));
  end
  median(vr);
  min(vr);
end

bag_id = x.ident.milbag; % 2002x1 each instance belongs to a bag
class = x.nlab; % 2002x1 each instance has a class label
n_bag = length(unique(bag_id));  % 58 bags with unique bag ids

%% Transform the data using PCA

[~,score,latent,~,explained] = pca(data);

save_to_base(1)
figure(2), subplot(2,1,1)
plot(1:100,latent(1:100)) % Kandemir uses the 100 first components
xlabel(sum(explained(1:100)))
title('Scree plots')
drawnow

% From the scree plot, it is apparent that many of those don't contribute
% We therefore have closer look at the first 10
subplot(2,1,2)
plot(1:10,latent(1:10)) 
xlabel(sum(explained(1:10)))
% And it seems like the first 5 or so suffice. 
% The xlabels show the proportion of variance explained by the 100 and 10
% first components, respectively. 

D = 5; % Based on the investigations, we choose dimension ? 
data = score(:,1:D);
data = data - mean(data);
data = data./sqrt(var(data));
% The bags are identified from the bag id
bag_class = zeros(1,n_bag);
size_bags = zeros(1,n_bag);
start = zeros(1,n_bag);
stop = zeros(1,n_bag);
for j = 1: n_bag
  start(j) = find(bag_id == j,1);
  stop(j) = find(bag_id == j,1,'last');
  size_bags(j) = stop(j)-start(j)+1;
  bag_class(j) = class(start(j));
end

%% I suspect that the number of patches in each image is correlated to the
% class label. 

[mean(size_bags(bag_class==1)) mean(size_bags(bag_class==2));
sqrt(var(size_bags(bag_class==1))) sqrt(var(size_bags(bag_class==2)))];

[~,eq_p] = ttest2(size_bags(bag_class==1),size_bags(bag_class==2),'Vartype','equal');
[~,uneq_p] = ttest2(size_bags(bag_class==1),size_bags(bag_class==2),'Vartype','unequal');

[eq_p uneq_p]; % I was right

%% 

min_length_bag = min(size_bags);

n_neg = sum(bag_class == 1); % Number of negative bags
n_pos = sum(bag_class == 2); % Number of positive bags
neg_idx = find(bag_class==1);
pos_idx = find(bag_class==2);

% These are my bags. I can either make all bags having the same number of
% instances by random sampling, or I can ignore this. 
x_bags = cell(D,n_bag);
for dim = 1: D   
  for j = 1: n_bag
    bag = data(start(j):stop(j),1:dim);
    x_bags{dim,j} = bag;  % datasample(bag,min_length_bag,'Replace',false); % 
  end
end

%%     
% Parameters for the EM-algorithm:  
precision = 'high'

if strcmp(precision,'low')
  maxiter = 100;  % Maximum number of iterations. 
  reps = 2; % Number of repetitions. For each rep, a new starting point. 
elseif strcmp(precision,'high')
  maxiter = 1000;
  reps = 10;
end
reg = 1e-6; % Avoiding non-invertible matrices
probtol = 1e-8; % Stopping criterion
start = 'randSample'; % 'plus' % Starting points. 
covtype = 'diagonal';
EMparam{1} = [maxiter reps reg probtol];
EMparam{2} = start; 
EMparam{3} = covtype;
criterion = 'AIC';
%%

% Parameters for distribution estimation
Kmin_class = 1; 
if strcmp(precision,'low')
  Kmax_class = 5; 
elseif strcmp(precision, 'high')
  Kmax_class = 20;
end
% No more than 5 is needed. neg_distr = 3 (4), pos_distr = 3(4)
Kmin_bag = 1; 
if strcmp(precision,'low')
  Kmax_bag = 5; %ceil(max(size_bags)/5)
elseif strcmp(precision,'high')
  Kmax_bag = 20;
end
K = [Kmin_class Kmax_class;
     Kmin_bag Kmax_bag];

bagfit = 1
bagfile = strcat('bags_distr_all_',precision);
if bagfit
  bags_distr = cell(D,n_bag,Kmax_bag);
  for dim = 1: D   
    % Fit a Gaussian mixture model to each of the bags
    for j = 1: n_bag
      [dim j];
      obj = gaussian_mixture(Kmin_bag,Kmax_bag,x_bags{dim,j},EMparam);
      bags_distr(dim,j,:) = obj;
    end
  end
  save(bagfile,'bags_distr')
else
  load(bagfile,'bags_distr')
end

rng('default') % For reproducibility in case bag = 0

F = 4; % 4 as in the paper. Performance does not increase for other F's
T = 10; % Number of repetitions for CV randomisation
AUC_ROC = zeros(3,D,T);

classfit = 1
for dim = 1: D
  for t = 1: T % the randomization might have an influence     
    % We use F-fold, stratified cross-validation. 
    n_ind = crossvalind('Kfold',n_neg,F);
    p_ind = crossvalind('Kfold',n_pos,F);
  
    fold = cell(1,F);  
    for f = 1: F
      fold{f} = [neg_idx(n_ind==f) pos_idx(p_ind==f)];
    end
  
    bag2class_div = zeros(4,n_bag);
    for f = 1: F % The folds of cross-validation
      pdf_file = strcat('class_pdf_file_low_prec_F',num2str(F),'_T',num2str(t),'_D',num2str(dim));
      x_neg = []; % The negative class
      x_pos = []; % The positive class
      
      balance = 1;
      if balance % Balance the two classes  
        n_fold_class = length(setdiff(pos_idx,fold{f})); % # bags in the positive class
        j_neg = datasample(setdiff(neg_idx,fold{f}),n_fold_class,'Replace',false); 
      else
        j_neg = setdiff(neg_idx,fold{f});
      end
      
      for j = j_neg 
        x_neg = [x_neg; x_bags{dim,j}];
      end      

      for j = setdiff(pos_idx,fold{f})
        x_pos = [x_pos; x_bags{dim,j}];
      end        
      
      % Take a look to see if they differ
      min_n_class = min(size(x_neg,1),size(x_pos,1)); % for unbalanced classes
      figure(3), hist([x_neg(1:min_n_class,dim) x_pos(1:min_n_class,dim)]), legend('Neg','Pos')
      drawnow
    
      %% Fit the distribution to the classes
      if classfit
        % Default: diagonal covariance matrix. Makes sense since we are
        % dealing with principal components. 
        % Fit a Gaussian mixture model to all negative instances
        neg_distr = gaussian_mixture(Kmin_class,Kmax_class,x_neg,EMparam);
      
        % Fit a Gaussian mixture model to all positive instances
        pos_distr = gaussian_mixture(Kmin_class,Kmax_class,x_pos,EMparam);
      
        save(pdf_file,'neg_distr','pos_distr')
      end
       
      bag2class_div(:,fold{f}) = breast_mil_div(fold{f}, bagfile, pdf_file, size_bags, bag_class, K, criterion, precision);    
    end
    
    condI_pos  = bag2class_div(1,:);
    condI_neg  = bag2class_div(2,:);
    I_neg = bag2class_div(3,:);
    I_pos = bag2class_div(4,:);
    y_test = bag_class;    
    ratioI = I_neg./I_pos; % I_pos./I_neg; %
  
    n_test = length(y_test);
    % Box-Cox for illustration
    [t_condI_pos,~] = boxcox((condI_pos-min(condI_pos)+eps)');
    [t_condI_neg,~] = boxcox((condI_neg-min(condI_neg)+eps)');
    [t_ratioI,~] = boxcox((ratioI-min(ratioI)+eps)');

    clf(figure(6)), subplot(1,3,1), hold on, title('Bag-to-class divergence')
    plot(-1,median(t_condI_pos),'.g'),plot(1,median(t_condI_pos),'.m')
    for j = 1: n_test
      if y_test(j) < 2
          plot(-0.25,t_condI_pos(j),'.g','MarkerSize',10)
      else
          plot(0.25,t_condI_pos(j),'.m','MarkerSize',10)
      end
    end

    subplot(1,3,2), hold on, title('Bag-to-class divergence (neg)')
    plot(-1,median(t_condI_neg),'.g'),plot(1,median(t_condI_neg),'.m')
    for j = 1: n_test
      if y_test(j) < 2
        plot(-0.25,t_condI_neg(j),'.g','MarkerSize',10)
      else
        plot(0.25,t_condI_neg(j),'.m','MarkerSize',10)
      end
    end

    subplot(1,3,3), hold on, title('Kullback-Leibler ratio')
    plot(-1,median(t_ratioI),'.g'),plot(1,median(t_ratioI),'.m')
    for j = 1: n_test
      if y_test(j) < 2 
        plot(-0.25,t_ratioI(j),'.g','MarkerSize',10)
      else
        plot(0.25,t_ratioI(j),'.m','MarkerSize',10)
      end
    end
    drawnow

    % Classification. Simple threshold.

    fig_nr = 7;
    AUC_ROC(:,dim,t) = classification_divergences(condI_pos, condI_neg, ratioI, bag_class, fig_nr, [1 0 1]);
  end
  AUC_ROC(:,dim,:);
 %{ 
[h,p,ci,stats] = ttest(squeeze(AUC_ROC(1,dim,:)));
  ci
  [h,p,ci,stats] = ttest(squeeze(AUC_ROC(3,dim,:)));
  ci
  
  
  [h,p,ci,stats] = ttest(squeeze(AUC_ROC(1,dim,:)),0.9);
  p
  ci
  [h,p,ci,stats] = ttest(squeeze(AUC_ROC(3,dim,:)),0.9);
  p
  ci
  %}
  meen = mean(AUC_ROC(:,dim,:),3)
  save_to_base(1)
end

ROC_file = strcat('AUC_ROC_F',num2str(F),'_',criterion,precision)
save(ROC_file,'AUC_ROC')
