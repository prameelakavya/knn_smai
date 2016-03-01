function labels = cvknn(x_test, x_train, y_train, categories, k)
%labels = [];
if size(x_test, 2) ~= size(x_train, 2)
    error('Dimensions of classifiee vectors and prototype vectors do not match.');
end
d = cveucdist(x_test', x_train');
[r,index] = sort(d,2);

for i = 1 : size(x_test,1)
    l = index(i,:);
    near_classes = [];
    for j = 1 : k
        m = l(j);
        near_classes = [near_classes; y_train(m,1)];
    end
    final = -1;
    for t = 1 : size(categories,1)
        if iscell(categories)
            categ = categories{t};
        else
            categ = categories(t);
        end
        if final ~= max(final, nnz(ismember(near_classes, categ)))
            final = max(final, nnz(ismember(near_classes, categ)));
            final_label = categ;
        end
    end
    if iscell(categories)
        labels{i,1} = final_label;
    else
        labels(i,1) = final_label;
    end
end

