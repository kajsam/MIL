function b2c_div = breast_mil_div(test_set, bag_pdfs, class_pdfs, size_bags, bag_class, K, criterion, precision)

% Calculates the conditional divergences, and the divergence ratio

% Requires bag_to_class_divergence.m, classification_divergences.m

% Input: bag_pdfs    - file with pdfs for each bag
%        class_pdfs  - file with pdfs for each class
%        y_test      - file containing class labels
%        K           - min/max number of components

addpath('/Volumes/kam025/Documents/MATLAB/MI_Divergence')

% load the class GMMs. Find minimum BIC or AIC.     
load(class_pdfs,'neg_distr','pos_distr')
dim = length(neg_distr{1}.mu);
Kmin = K(1,1);  Kmax = K(1,2);

% Negative class (benign lesions)
ABIC = zeros(1,Kmax-Kmin+1);
if strcmp(criterion,'BIC')
  for k = Kmin: Kmax
    ABIC(k-Kmin+1) = neg_distr{k}.BIC;
  end
elseif strcmp(criterion,'AIC')
  for k = Kmin: Kmax
    ABIC(k-Kmin+1) = neg_distr{k}.AIC;
  end
end
[~,k_ABICneg] = min(ABIC);
k_ABICneg = Kmin-1+k_ABICneg;
neg_distr = neg_distr{k_ABICneg};

% Positive class (malignant lesions)
ABIC = zeros(1,Kmax-Kmin+1);
if strcmp(criterion,'BIC')
  for k = Kmin: Kmax
    ABIC(k-Kmin+1) = pos_distr{k}.BIC;
  end
elseif strcmp(criterion,'AIC')
  for k = Kmin: Kmax
    ABIC(k-Kmin+1) = pos_distr{k}.AIC;
  end 
end

[~,k_ABICpos] = min(ABIC);
k_ABICpos = Kmin-1+k_ABICpos;
pos_distr = pos_distr{k_ABICpos};

[k_ABICneg k_ABICpos]

% load the bag GMMs. Find minimum BIC AIC. 
load(bag_pdfs,'bags_distr')
distr = bags_distr(dim,test_set,:);
size_bags = size_bags(test_set);
Kmin = K(2,1); Kmax = K(2,2);

ABIC = zeros(1,Kmax-Kmin+1); k_ABIC = zeros(1,Kmax-Kmin+1);
obj = cell(1, size(distr,1));
if strcmp(criterion,'BIC')
  for i = 1: size(distr,2)
    % Kmax = ceil(size_bags(i)/10); % I had an idea on Kmax for each bag, but
    % it would possibly favor the bag-size-class-label-correlation
    for k = Kmin: Kmax
      ABIC(k) = distr{1,i,k}.BIC;
    end
    [~,k_ABIC(i)] = min(ABIC(Kmin:Kmax));
    k_ABIC(i) = Kmin-1+k_ABIC(i);    
    obj{i} = distr{1,i,k_ABIC(i)};
  end
elseif strcmp(criterion,'AIC')
  for i = 1: size(distr,2)
    % Kmax = ceil(size_bags(i)/10);
    for k = Kmin: Kmax
      save_to_base(1)
      ABIC(k) = distr{1,i,k}.AIC;
    end
    [~,k_ABIC(i)] = min(ABIC(Kmin:Kmax));
    k_ABIC(i) = Kmin-1+k_ABIC(i);
    obj{i} = distr{1,i,k_ABIC(i)};
  end
end

k_ABIC(1:5);
bags_distr = obj;
  
% Importance sampling
Mu_imp = [neg_distr.mu; pos_distr.mu];
Sigma_imp = cat(3,neg_distr.Sigma, pos_distr.Sigma);
% I weight neg and pos classes equally, else, I might get unwanted bias
% when pos << neg
P_imp = [neg_distr.ComponentProportion pos_distr.ComponentProportion]; 
imp_distr = gmdistribution(Mu_imp,Sigma_imp,P_imp);

if strcmp(precision,'low')
  n = 100000; % Number of sample points Default: 100 000. This takes no time at all
elseif strcmp(precision,'high')
  n = 1000000;
end
X = random(imp_distr,n);
imp = pdf(imp_distr,X);

if dim == 0 % Displaying fitted distributions. Only for 1st dimension
  bag_class = bag_class(test_set);
  neg_idx = find(bag_class == 1);
  pos_idx = find(bag_class == 2);
  mn = min(X); mx = max(X); stp = (mx-mn)/1000;
  x = (mn:stp:mx)';

  figure(4), hold on, % Shows fitted distr
  title('Fitted distributions') 
  y = pdf(neg_distr, x);
  plot(x,y,'b')
  y = pdf(pos_distr, x);
  plot(x,y,'r')
  y = pdf(imp_distr, x);
  plot(x,y,'m')
  legend('Negative class','Positive class','Importance','Location','NE')
        
  clf(figure(5)), hold on, title('Fitted distributions') % Shows fitted distr
  y = pdf(neg_distr, x);
  plot(x,y,'b')
  y = pdf(pos_distr, x);
  plot(x,y,'r')
  j_neg = neg_idx(1);
  for j = j_neg
    y = pdf(bags_distr{j}, x);  
    plot(x,y,'g')
  end
  j_pos = pos_idx(1);
  for j = j_pos
    y = pdf(bags_distr{j}, x);  
    plot(x,y,'m')
  end
  legend('Negative class','Positive class','Bags','Location','NE')
end  

b2c_div = bag_to_class_divergence(neg_distr,pos_distr,bags_distr,imp,X);
save_to_base(1)
