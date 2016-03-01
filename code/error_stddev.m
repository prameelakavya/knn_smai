
% for i = 2 : 5
%     for k = 1 : 5
%         [a,b,c] = knn_accuracy(x_iris, y_iris, categories_iris, i, 10, k);
%         mean_error_iris = [mean_error_iris ; a];
%         stddev_iris = [stddev_iris ; c];
%     end
% end

%  for i = 2 : 5
%      for k = 1 : 5
%          [a,b,c] = knn_accuracy(x_car, y_car, categories_car, i, 10, k);
%          mean_error_car = [mean_error_car ; a];
%          stddev_car = [stddev_car ; c];
%      end
%  end
% 
% for i = 2 : 5
%     for k = 1 : 5
%         [a,b,c] = knn_accuracy(x_wine, y_wine, categories_wine, i, 10, k);
%         mean_error_wine = [mean_error_wine ; a];
%         stddev_wine = [stddev_wine ; c];
%     end
% end
% 
for i = 2 : 5
    for k = 1 : 5
        [a,b,c] = knn_accuracy(x_public, y_public, categories_public, i, 10, k);
        mean_error_public = [mean_error_public ; a];
        stddev_public = [stddev_public ; c];
    end
end