function I = bag_to_class_divergence(neg_distr,pos_distr,bags_distr,imp,X)

% Input:    neg_distr   - the pdf of the negative class
%           pos_distr   - the pdf of the positive class
%           bags_distr   - the pdf of the bags
%           imp         - the pdf of the importance sampling distribution,
%                         evaluated at X
%           X           - samples from the importance distribution
%           fold        - when using cross-validation

% Approximations of integrals by importance sampling

% Calculate the bag-to-class conditional divergence, and the Kullbak-Leibler 
% ratio as a reference.

const = 1; % Display reasons

n_bags = length(bags_distr);
condI_pos = zeros(1,n_bags); % conditioning on positive class
condI_neg = zeros(1,n_bags); % conditioning on negative class
I_neg = zeros(1,n_bags); % from bag to negative class
I_pos = zeros(1,n_bags); % from bag to positive class
% pdf of negative and positive class distributions
p_neg = const.*pdf(neg_distr,X);
p_pos = const.*pdf(pos_distr,X);

n = length(X);

for j = 1:n_bags
  obj = bags_distr{j};
  p_bag = const.*pdf(obj,X);        % bag pdf
      
  % Calculate each term
  cond_terms_pos = p_pos.*p_bag.*log(p_bag./p_neg)./imp; %(p_pos./p_neg).*p_bag.*log(p_bag./p_neg)./imp; % 
  cond_terms_neg = p_neg.*p_bag.*log(p_bag./p_pos)./imp; %(p_neg./p_pos).*p_bag.*log(p_bag./p_pos)./imp; % 
  % Alt: p_pos.*p_bag.*log(p_bag./p_neg)./imp; % In case of instability  
  terms_neg = p_bag.*log(p_bag./p_neg)./imp;
  terms_pos = p_bag.*log(p_bag./p_pos)./imp;

  % 0log0 = 0
  cond_terms_pos(p_bag == 0) = 0; cond_terms_neg(p_bag == 0) = 0; 
  cond_terms_pos(p_pos == 0) = 0; cond_terms_neg(p_neg == 0) = 0; 
  terms_neg(p_bag == 0) = 0; terms_pos(p_bag == 0) = 0;
  
  % Checkin here for any NaN
  if any(isnan([cond_terms_pos cond_terms_neg terms_neg terms_pos]))
    print('NaNs - somethings wrong')
    return 
  end
  
  % Only terms where p_pos>p_neg contributes to I_bc
  cond_n_pos = n;  %   sum(p_neg<p_pos); % 
  % cond_terms_pos(p_pos <= p_neg) = 0;  
  cond_n_neg = n; %  sum(p_neg>p_pos); % 
  % cond_terms_neg(p_neg <= p_pos) = 0;  
  
  condI_pos(j)  = sum(cond_terms_pos)/cond_n_pos;
  condI_neg(j)  = sum(cond_terms_neg)/cond_n_neg;
  I_neg(j) = sum(terms_neg)/n;    
  I_pos(j) = sum(terms_pos)/n;
end

I = [condI_pos; condI_neg; I_neg; I_pos];
