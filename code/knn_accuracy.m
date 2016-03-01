function [classifier_error, classifier_variance, classifier_stddev]  = knn_accuracy(x, y, categories, numFolds, testruns, k)

% get the number of vectors corresponding to each category ----------------
numCats = length(categories);

%'vecsPerCat will store the number of input vectors belonging to each category.
vecsPerCat = zeros(numCats, 1); % numCats x 1 is the order of the matrix

for i = 1 : numCats
    
    % Get the ith category; store the category value in column 1.
    categ = categories(i);
    
    % Count the number of input vectors with that category.
    vecsPerCat(i, 1) = sum(ismember(y , categ)); 
end

%--------------------------------------------------------------------------

% get the segregated parts of each category samples------------------------

a = size(vecsPerCat);
numCats = a(1,1);
 for i = 1 : numCats
    
    % Get the number of vectors for this category;
    numVecs = vecsPerCat(i, 1);
    
    % Verify that there are at least 'numFolds' samples.
    if (numVecs < numFolds)
        disp('ERROR! Each category must have at least' + numFolds + 'samples.');
    end
 end

% foldSizes will be a matrix holding the number of vectors to place in each fold
% for each category. The number of folds may not divide evenly into the number
% of vectors, so we need to distribute the remainder.
foldSizes = zeros(numCats, numFolds);

% For each category...
for i = 1 : numCats
    
    % Get the the number of vectors for this category.
    numVecs = vecsPerCat(i, 1);
            
    % For each of the ten folds...
    for fold = 1 : numFolds
        
        % Divide the remaining number of vectors by the remaining number of folds.
        foldSize = ceil(numVecs / (numFolds - fold + 1));
        
        % Store the fold size.
        foldSizes(i, fold) = foldSize;
        
        % Update the number of remaining vectors for this category.
        numVecs = numVecs - foldSize;
    end
end
% Verify the fold sizes sum up correctly.
if (any(sum(foldSizes, 2) ~= vecsPerCat))
    disp('ERROR! The sum of fold sizes did not equal the number of category vectors.');
end
%--------------------------------------------------------------------------

% randomly arrange x and y and then arrange them according to their categ--

% Get the total number of input vectors.
totalVecs = size(x,1);
E = 0;
E_matrix = [];
variance = 0;
classifier_stddev = 0;

for t = 1 : testruns   % for loop for test runs of cross validation--------
% Get a random order of the indeces.
randOrder = randperm(totalVecs);

% Sort the vectors and categories with the random order.
randVecs = x(randOrder, :);
randCats = y(randOrder, :);

x_sorted = [];
y_sorted = [];

% Re-group the vectors according to category.
for i = 1 : size(categories,1)
    
    % Get the category value.
    categ = categories(i);
    
    % Select all of the vectors for this category.
    catVecs = randVecs(ismember(randCats , categ), :); % this works bcoz of same random order for both x and y
    catCats = randCats(ismember(randCats , categ), :);
   
    % Append the vectors for this category.
    x_sorted = [x_sorted; catVecs];
    y_sorted = [y_sorted; catCats];
end

%--------------------------------------------------------------------------

x_test = [];
y_test = [];
x_train = [];
y_train = [];
catStart = 1;
no_of_misclass = 0;
no_of_tests = 0;

for blockNumber = 1 : numFolds
% For each category...
    for catIndex = 1 : size(categories,1)

        % Get the list of fold sizes for this category as a column vector.
        catFoldSizes = transpose(foldSizes(catIndex, :));
    
        % Set the starting index of the first fold for this category.
        foldStart = catStart;
    
        % For each fold...
        for foldIndex = 1 : numFolds
        
            % Compute the index of the last vector in this fold.
            foldEnd = foldStart + catFoldSizes(foldIndex) - 1;
        
            % Select all of the vectors in this fold.
            foldVectors = x_sorted(foldStart : foldEnd, :);
            foldCats = y_sorted(foldStart : foldEnd, :);
        
            % If this fold is to be used for validation in this round...
            if (foldIndex == blockNumber)
                % Append the vectors to the validation set.
                x_test = [x_test; foldVectors];
                y_test = [y_test; foldCats];
            % Otherwise, use the fold for training.
            else
                % Append the vectors to the training set.
                x_train = [x_train; foldVectors];
                y_train = [y_train; foldCats];
            end
            
            % Update the starting index of the next fold.
            foldStart = foldEnd + 1;
        end
    
        % Set the starting index of the next category.
        catStart = catStart + vecsPerCat(catIndex);   
    end
   
    %---training and testing and computing error based on KNN--------------

    %mdl = ClassificationKNN.fit(x_train,y_train,'NumNeighbors',2);
    %label = predict(mdl,x_test);
    label = cvknn(x_test, x_train, y_train, categories, k);
    no_of_tests = no_of_tests + size(label,1);
    for u = 1 : size(label,1)
        if iscell(categories)
            if ~strcmp(label{u,1}, y_test{u,1})
                 no_of_misclass = no_of_misclass + 1;
            end
        else
            if ~(label(u,1) == y_test(u,1))
                no_of_misclass = no_of_misclass + 1;
            end
        end
    end
    %no_of_misclass = no_of_misclass + nnz(~ismember(label', y_test));
    x_test = [];
    y_test = [];
    x_train = [];
    y_train = [];
    catStart = 1;
    
end
no_of_tests;
no_of_misclass;
E = E + no_of_misclass/size(x, 1);
E_matrix = [E_matrix ; E];

end

error = E/testruns;
classifier_error = error;
variance = 0;

for i = 1 : size(E_matrix,1)
    variance = variance + (error - E_matrix(i))^2;
end

classifier_variance = (variance/(testruns - 1)) ;
classifier_stddev = sqrt(classifier_variance);











