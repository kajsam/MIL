function distr = gaussian_mixture(K_min,K_max,x,EMparam)

% rng('default') WTF?

maxiter = EMparam{1}(1);
reps = EMparam{1}(2);
reg  = EMparam{1}(3);
probtol = EMparam{1}(4);
start = EMparam{2};
covtype = EMparam{3};
options = statset('MaxIter',maxiter); 

fit_distr = cell(1,K_max-K_min+1);
BIC = zeros(1,K_max-K_min+1);
for k = K_min: K_max
  fit_distr{k} = fitgmdist(x,k,'Regularize',reg,'Options',options,'Replicates',reps, ...
                               'ProbabilityTolerance',probtol,'Start',start,...
                               'CovarianceType',covtype);
  BIC(k) = fit_distr{k}.BIC;
end
  
k_new = BIC==min(BIC);
find(k_new);
distr = fit_distr;