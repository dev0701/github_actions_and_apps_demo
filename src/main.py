import import_data
import perceptron
import predict
import KNN
import KNN_new
import kfoldcv
import pickle
# set which algorithm to run 0 for perceptron, 1 for knn (default set to perceptron):
algorithm = 1
# learning rate for perceptron
learning_rate = 1
# set number of trials (default set to 1)
num_runs = 1
# set validation method (1 for K-fold CV)
validation_method = 0
# set target district 1 - 32
target_district = 1
# set district for negative label 1 - 32 excluding target_district
neg_district = 2
# number of nearest neighbors
k = 1
#csv path for data
csv = "data-set-no-grade.csv"
#number of folds in K-fold cross validation
k_cross_val = 5
###################Training/Validating/Testing##########################
if(validation_method == 0):
    # Perceptron
    if algorithm == 0 :
        total_accuracy = 0
        total_perceptron_true_positive = 0
        total_perceptron_true_negative = 0
        total_perceptron_false_positive = 0
        total_perceptron_false_negative = 0
        total_accuracy_validation = 0
        for l in range (num_runs):
            perceptron_correct = 0
            perceptron_true_positive = 0
            perceptron_true_negative = 0
            perceptron_false_positive = 0
            perceptron_false_negative = 0
            # split data for training/testing
            #f = open('store.pckl', 'rb')
            #X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(f)
            X_train, y_train, X_test, y_test, X_val, y_val = import_data.run("data-set-no-grade.csv", 2,4, 40)
            #f = open('store.pckl', 'wb')
            #pickle.dump([X_train, y_train, X_test, y_test, X_val, y_val], f)
            #f.close()
            # perceptron training
            theta, iter = perceptron.run(200, X_train, y_train, learning_rate)
            # perceptron testing
            for i in range(len(y_test)):
                print("Prediction: " + str(predict.run(theta, X_test[i])))
                print("Actual: " + str(y_test[i]))

                if predict.run(theta, X_test[i]) == y_test[i] :
                    perceptron_correct = perceptron_correct + 1
                    if y_test[i] == 1:
                        perceptron_true_positive = perceptron_true_positive + 1
                    else:
                        perceptron_true_negative = perceptron_true_negative + 1
                elif y_test[i] != 1:
                    perceptron_false_positive = perceptron_false_positive + 1
                else:
                    perceptron_false_negative = perceptron_false_negative + 1
            total_accuracy = total_accuracy + (perceptron_correct/len(y_test))
            total_perceptron_true_positive = total_perceptron_true_positive + perceptron_true_positive
            total_perceptron_true_negative = total_perceptron_true_negative + perceptron_true_negative
            total_perceptron_false_positive = total_perceptron_false_positive + perceptron_false_positive
            total_perceptron_false_negative = total_perceptron_false_negative + perceptron_false_negative
            perceptron_correct = 0
            perceptron_true_positive = 0
            perceptron_true_negative = 0
            perceptron_false_positive = 0
            perceptron_false_negative = 0
            for i in range(len(y_val)):
                print("Prediction: " + str(predict.run(theta, X_val[i])))
                print("Actual: " + str(y_val[i]))
                # print("Theta: " + str(theta))
                if predict.run(theta, X_val[i]) == y_val[i] :
                    perceptron_correct = perceptron_correct + 1
                    if y_val[i] == 1:
                        perceptron_true_positive = perceptron_true_positive + 1
                    else:
                        perceptron_true_negative = perceptron_true_negative + 1
                elif y_val[i] != 1:
                    perceptron_false_positive = perceptron_false_positive + 1
                else:
                    perceptron_false_negative = perceptron_false_negative + 1
            total_accuracy_validation = total_accuracy_validation + (perceptron_correct/len(y_val))
            total_perceptron_true_positive = total_perceptron_true_positive + perceptron_true_positive
            total_perceptron_true_negative = total_perceptron_true_negative + perceptron_true_negative
            total_perceptron_false_positive = total_perceptron_false_positive + perceptron_false_positive
            total_perceptron_false_negative = total_perceptron_false_negative + perceptron_false_negative
        average_accuracy = total_accuracy/num_runs
        average_accuracy_val = total_accuracy_validation/num_runs
        print("true positive: " + str(total_perceptron_true_positive))
        print("Average Perceptron validation accuracy across " + str(num_runs) + " runs was: " + str(average_accuracy_val * 100) + "%" )
        print("Average Perceptron testing accuracy across " + str(num_runs) + " runs was: " + str(average_accuracy * 100) + "%" )
        print("True positive rate for perceptron across " + str(num_runs) + " runs was: " + str(total_perceptron_true_positive/(total_perceptron_true_positive+total_perceptron_false_negative)) )
        print("False positive rate for perceptron across " + str(num_runs) + " runs was: " + str(total_perceptron_false_positive/(total_perceptron_false_positive+total_perceptron_true_negative)) )

    # KNN with variable k
    if algorithm == 1:
        X_train, y_train, X_test, y_test, X_val, y_val = import_data.run("data-set-no-grade.csv", 2,3, 40)
        print(KNN_new.run(k, X_train, y_train, X_test, y_test, X_val,
            y_val))

    # KNN with a single k
    if algorithm == 2 :
        total_accuracy = 0
        total_knn_true_positive = 0
        total_knn_true_negative = 0
        total_knn_false_positive = 0
        total_knn_false_negative = 0
        total_accuracy_validation = 0
        for l in range (num_runs):
            knn_correct = 0
            knn_true_positive = 0
            knn_true_negative = 0
            knn_false_positive = 0
            knn_false_negative = 0
            # split data for training/testing
            f = open('store.pckl', 'rb')
            X_train, y_train, X_test, y_test, X_val, y_val = pickle.load(f)
            #X_train, y_train, X_test, y_test, X_val, y_val = import_data.run("data-set-no-grade.csv", 2,3, 40)
            #f = open('store.pckl', 'wb')
            #pickle.dump([X_train, y_train, X_test, y_test, X_val, y_val], f)
            f.close()
            for i in range(len(y_test)):
                print("Prediction: " + str(KNN.run(X_train, y_train, X_test[i])))
                print("Actual: " + str(y_test[i]))
                if KNN.run(X_train, y_train, X_test[i]) == y_test[i] :
                    knn_correct = knn_correct + 1
                    if y_test[i] == 1:
                        knn_true_positive = knn_true_positive + 1
                    else:
                        knn_true_negative = knn_true_negative + 1
                elif y_test[i] != 1:
                    knn_false_positive = knn_false_positive + 1
                else:
                    knn_false_negative = knn_false_negative + 1
            total_accuracy = total_accuracy + (knn_correct/len(y_test))
            total_knn_true_positive = total_knn_true_positive + knn_true_positive
            total_knn_true_negative = total_knn_true_negative + knn_true_negative
            total_knn_false_positive = total_knn_false_positive + knn_false_positive
            total_knn_false_negative = total_knn_false_negative + knn_false_negative
            knn_correct = 0
            knn_true_positive = 0
            knn_true_negative = 0
            knn_false_positive = 0
            knn_false_negative = 0
            for i in range(len(y_val)):
                print("Prediction: " + str(KNN.run(X_train, y_train, X_val[i])))
                print("Actual: " + str(y_val[i]))
                if KNN.run(X_train, y_train, X_val[i]) == y_val[i] :
                    knn_correct = knn_correct + 1
                    if y_val[i] == 1:
                        knn_true_positive = knn_true_positive + 1
                    else:
                        knn_true_negative = knn_true_negative + 1
                elif y_val[i] != 1:
                    knn_false_positive = knn_false_positive + 1
                else:
                    knn_false_negative = knn_false_negative + 1
            total_accuracy_validation = total_accuracy_validation + (knn_correct/len(y_val))
            total_knn_true_positive = total_knn_true_positive + knn_true_positive
            total_knn_true_negative = total_knn_true_negative + knn_true_negative
            total_knn_false_positive = total_knn_false_positive + knn_false_positive
            total_knn_false_negative = total_knn_false_negative + knn_false_negative
        average_accuracy = total_accuracy/num_runs
        average_accuracy_val = total_accuracy_validation/num_runs
        print("Average KNN validation accuracy across " + str(num_runs) + " runs was: " + str(average_accuracy_val* 100) + "%" )
        print("Average KNN testing accuracy across " + str(num_runs) + " runs was: " + str(average_accuracy * 100) + "%" )
        print("True positive rate for KNN across " + str(num_runs) + " runs was: " + str(total_knn_true_positive/(total_knn_true_positive+total_knn_false_negative)) )
        print("False positive rate for KNN across " + str(num_runs) + " runs was: " + str(total_knn_false_positive/(total_knn_false_positive+total_knn_true_negative)) )
###############################END######################################
###################Training/Validating/Testing##########################

###################### K-Fold Cross Validation #########################
if(validation_method != 0):
    kfoldcv.run(csv, target_district, neg_district, k_cross_val, algorithm)
