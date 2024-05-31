import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.IntuitionisticFuzzySet import IntuitionisticFuzzySet


class G_Measure:
    def __init__(self, x_space_grid_size=1000, space_expansion=0.5):
        self.x_space_grid_size = x_space_grid_size
        self.space_expansion = space_expansion

    def fit(self, y_true, y_score=None, y_pred=None, considered_classes='all', aggregation_technique='average',
            aggregation_technique_params=None, data_estimation_method='kde',
            data_estimation_method_params=None, data_density_estimation_normalization=True, data_balance=None):
        """
        Parameters
        ----------
        y_true :
            True labels.

        y_score :
            Scores or labels obtained from a classifier. G Measure calculated by density plots.

        y_pred :
            Labels predictions obtained from a classifier. G Measure calculated by confusion matrix.

        considered_classes : {'all', list, tuple}, default='all'
            If 'all', all classes in the dataset are considered for processing.
            If a list or tuple, only the classes specified within are considered. Each element in the list or tuple should be a class label. It controls which classes are included during the training stage.

        aggregation_technique : {'average', 'weighted_average'}, default='average'
            Selects the aggregation technique used to calculate the overall G Measure from the single class G Measure results.
            'average':
                Overall G Measure is calculated as simple average of single class G Measure results.
            'weighted_average':
                Overall G Measure is calculated as weighted average of single class G Measure results.

        aggregation_technique_params : dict or None, default=None
            Parameters for the selected aggregation technique.

        data_estimation_method : {'kde', 'histogram'}, default='kde'
            Specifies the method used for estimating the data distribution from the given y_score.
            'kde':
                Kernel Density Estimation that uses a Gaussian kernel to estimate the probability density function of the y_score.
            'histogram':
                A histogram-based method that estimates the probability density function of the y_score.

        data_estimation_method_params : dict or None, default=None
            Parameters for the data estimation method.

        data_density_estimation_normalization : bool, default=True
            Determines whether to normalize the density estimation results.

        data_balance : dict or None, default=None
            A dictionary specifying the balancing factors for each class, where keys are class labels and values are the balancing weights.

        Returns
        -------
        None
        """

        if data_estimation_method_params is None:
            data_estimation_method_params = {}
        if considered_classes == 'all':
            y_true = y_true.copy()
            if y_score is not None:
                y_score = y_score.copy()
            if y_pred is not None:
                y_pred = y_pred.copy()
        elif isinstance(considered_classes, (list, tuple)) and all(
                isinstance(item, (int, float)) for item in considered_classes):
            mask = np.isin(y_true, considered_classes)
            y_true = y_true[mask]
            if y_score is not None:
                y_score = y_score[mask]
            if y_pred is not None:
                y_pred = y_pred[mask]

            missing_classes = set(considered_classes) - set(y_true)
            if missing_classes:
                raise ValueError(f"Missing considered classes in y_true: {missing_classes}.")
        else:
            raise ValueError("Incorrect G Measure classes selection.")

        if len(set(y_true)) < 2:
            raise ValueError("Unique classes number less than 2.")

        self.unique_classes = sorted(set(y_true))
        self.kernels = {}
        if y_score is not None and y_pred is None:
            difference = max(y_score) - min(y_score)
            self.x_space = np.linspace(min(y_score) - self.space_expansion * difference,
                                       max(y_score) + self.space_expansion * difference, self.x_space_grid_size)
            for unique_class in self.unique_classes:
                if data_estimation_method == 'kde':
                    kernel = scipy.stats.gaussian_kde(y_score[y_true == unique_class])
                    self.kernels[unique_class] = kernel(self.x_space)
                elif data_estimation_method == 'histogram':
                    counts, bin_edges = np.histogram(y_score[y_true == unique_class],
                                                     bins=data_estimation_method_params['bins'], density=True)
                    self.kernels[unique_class] = np.zeros_like(self.x_space)
                    idx = np.digitize(self.x_space, bin_edges, right=True)
                    valid_bins = (idx > 0) & (idx <= len(counts))
                    self.kernels[unique_class][valid_bins] = counts[idx[valid_bins] - 1]
                    area = np.trapz(self.kernels[unique_class], self.x_space)
                    self.kernels[unique_class] /= area
                else:
                    raise ValueError("Incorrect G Measure data estimation method.")

                if data_density_estimation_normalization == False:
                    scale_factors = float(len(y_score[y_true == unique_class]))
                else:
                    scale_factors = 1.0
                if data_balance != None:
                    scale_factors = scale_factors * data_balance[unique_class]

                self.kernels[unique_class] = self.kernels[unique_class] * scale_factors
        elif y_score is None and y_pred is not None:
            y_true = y_true.to_numpy()
            matrix = confusion_matrix(y_true, y_pred, labels=self.unique_classes)
            self.x_space = [str(category) for category in self.unique_classes]
            for index, unique_class in enumerate(self.unique_classes):
                if data_density_estimation_normalization == False:
                    scale_factors = 1.0
                else:
                    scale_factors = 1.0 / float(np.sum(matrix[index]))
                if data_balance != None:
                    scale_factors = scale_factors * data_balance[unique_class]

                self.kernels[unique_class] = matrix[index] * scale_factors
        elif y_score is not None and y_pred is not None:
            raise ValueError(
                f"y_score and y_pred are not None. It is unclear which data should be used for training purposes.")
        else:
            raise ValueError(f"y_score or y_pred must be not None.")

        self.other_kernels = {}
        self.intersection_kernels = {}
        for unique_class in self.unique_classes:
            other_kernels = [kernel for other_class_index, kernel in self.kernels.items() if
                             other_class_index != unique_class]
            self.other_kernels[unique_class] = np.max(other_kernels, axis=0)
            self.intersection_kernels[unique_class] = np.minimum(self.kernels[unique_class],
                                                                 self.other_kernels[unique_class])

        self.thresholds = []
        kernels_matrix = np.array([self.kernels[unique_class] for unique_class in self.unique_classes])
        dominant_classes_indices = np.argmax(kernels_matrix, axis=0)
        dominant_classes = [self.unique_classes[index] for index in dominant_classes_indices]

        current_class = dominant_classes[0]
        for i in range(1, len(dominant_classes)):
            if dominant_classes[i] != current_class:
                if y_score is not None and y_pred is None:
                    self.thresholds.append((current_class, (self.x_space[i - 1] + self.x_space[i]) / 2.0))
                else:
                    self.thresholds.append((current_class, self.x_space[i - 1]))
                current_class = dominant_classes[i]
        if y_score is not None and y_pred is None:
            self.thresholds.append((current_class, float('inf')))
        else:
            self.thresholds.append((current_class, self.x_space[i]))

        self.intersection_current_dominant_kernels = {}
        self.intersection_current_non_dominant_kernels = {}
        for unique_class in self.unique_classes:
            self.intersection_current_dominant_kernels[unique_class] = np.zeros_like(
                self.intersection_kernels[unique_class])
            dominant_mask = [dominant_class == unique_class for dominant_class in dominant_classes]
            self.intersection_current_dominant_kernels[unique_class][dominant_mask] = \
            self.intersection_kernels[unique_class][dominant_mask]

            self.intersection_current_non_dominant_kernels[unique_class] = np.zeros_like(
                self.intersection_kernels[unique_class])
            non_dominant_mask = [dominant_class != unique_class for dominant_class in dominant_classes]
            self.intersection_current_non_dominant_kernels[unique_class][non_dominant_mask] = \
                self.intersection_kernels[unique_class][non_dominant_mask]

        self.class_measure = {}
        for unique_class in self.unique_classes:
            degreeOfMembership = 1.0 - np.sum(self.intersection_kernels[unique_class]) / np.sum(
                self.kernels[unique_class])
            degreeOfNonMembership = np.sum(self.intersection_current_non_dominant_kernels[unique_class]) / np.sum(self.kernels[unique_class])
            # degreeOfUncertainty = np.sum(self.intersection_current_dominant_kernels[unique_class]) / np.sum(
            #     self.kernels[unique_class])
            self.class_measure[unique_class] = IntuitionisticFuzzySet(degreeOfMembership=degreeOfMembership,
                                                                      degreeOfNonMembership=degreeOfNonMembership)

        if aggregation_technique == 'average':
            degreeOfMembership = 0.0
            degreeOfNonMembership = 0.0
            for class_index in self.unique_classes:
                degreeOfMembership += self.class_measure[class_index].degreeOfMembership
                degreeOfNonMembership += self.class_measure[class_index].degreeOfNonMembership
            self.overall_measure = IntuitionisticFuzzySet(
                degreeOfMembership=degreeOfMembership / len(self.unique_classes),
                degreeOfNonMembership=degreeOfNonMembership / len(self.unique_classes))
        elif aggregation_technique == 'weighted_average':
            weights = aggregation_technique_params['weights']
            degreeOfMembership = 0.0
            degreeOfNonMembership = 0.0
            weights_sum = 0.0
            for class_index in self.unique_classes:
                weights_sum += weights[class_index]
                degreeOfMembership += self.class_measure[class_index].degreeOfMembership * weights[class_index]
                degreeOfNonMembership += self.class_measure[class_index].degreeOfNonMembership * weights[class_index]
            self.overall_measure = IntuitionisticFuzzySet(
                degreeOfMembership=degreeOfMembership / weights_sum,
                degreeOfNonMembership=degreeOfNonMembership / weights_sum)
        else:
            raise ValueError("Incorrect G Measure aggregation technique.")

    def predict(self, y_score):
        if self.thresholds is not None:
            predictions = []
            for score in y_score:
                for class_index, threshold in self.thresholds:
                    if isinstance(self.x_space[0], str):
                        if str(score) == threshold:
                            predictions.append(class_index)
                            break
                    else:
                        if score <= threshold:
                            predictions.append(class_index)
                            break
            return predictions
        else:
            raise ValueError("G measure model not fitted.")

    def print_class_measure(self):
        if self.class_measure is not None:
            print("############################################")
            print("####### G Measure - single class ###########")
            print("############################################")
            for class_index in self.unique_classes:
                print("Class = " + str(class_index))
                print(self.class_measure[class_index])
            print("############################################")
            print("############################################")
            print("############################################")
        else:
            raise ValueError("G measure model not fitted.")

    def print_overall_measure(self):
        if self.overall_measure is not None:
            print("############################################")
            print("######### G Measure - overall ##############")
            print("############################################")
            print(self.overall_measure)
            print("############################################")
            print("############################################")
            print("############################################")
        else:
            raise ValueError("G measure model not fitted.")

    def print_thresholds(self):
        if self.thresholds is not None:
            print("############################################")
            print("####### G Measure - thresholds #############")
            print("############################################")
            for class_index, threshold in self.thresholds:
                if isinstance(threshold, str):
                    print(f"G Measure prediction {class_index} from classifier prediction {threshold}")
                else:
                    print(f"G Measure prediction {class_index} from classifier threshold <= {threshold}")
            print("############################################")
            print("############################################")
            print("############################################")
        else:
            raise ValueError("G measure model not fitted.")

    def plot_class_chart(self, file_name='plot', title=True):
        bar_width = 0.25
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.unique_classes))]

        for unique_class in self.unique_classes:
            if isinstance(self.x_space[0], str) == True:
                plt.bar(self.x_space, np.maximum(self.kernels[unique_class], self.other_kernels[unique_class]),
                        width=bar_width, label='Other',
                        color='yellow', alpha=1.0, edgecolor='black', linewidth=2.0)
                plt.bar(self.x_space, self.kernels[unique_class], width=bar_width,
                        label='Membership (%0.2f)' % self.class_measure[unique_class].degreeOfMembership,
                        color='green', alpha=1.0, edgecolor='black', linewidth=2.0)
                plt.bar(self.x_space, self.intersection_current_non_dominant_kernels[unique_class], width=bar_width,
                        label='Non-membership (%0.2f)' % self.class_measure[unique_class].degreeOfNonMembership,
                        color='red', alpha=1.0, edgecolor='black', linewidth=2.0)
                plt.bar(self.x_space, self.intersection_current_dominant_kernels[unique_class], width=bar_width,
                        label='Uncertainty (%0.2f)' % self.class_measure[unique_class].degreeOfUncertainty,
                        color='black', alpha=1.0, edgecolor='black', linewidth=2.0)
            else:
                plt.plot(self.x_space, np.maximum(self.kernels[unique_class], self.other_kernels[unique_class]),
                         color='black')
                plt.fill_between(self.x_space, np.maximum(self.kernels[unique_class], self.other_kernels[unique_class]),
                                 color='yellow', alpha=1.0, label='Other')

                plt.plot(self.x_space, self.kernels[unique_class], color='black')
                plt.fill_between(self.x_space, self.kernels[unique_class], color='green', alpha=1.0,
                                 label='Membership (%0.2f)' % self.class_measure[unique_class].degreeOfMembership)

                plt.plot(self.x_space, self.intersection_current_non_dominant_kernels[unique_class], color='black')
                plt.fill_between(self.x_space, self.intersection_current_non_dominant_kernels[unique_class],
                                 color='red', alpha=1.0, label='Non-membership (%0.2f)' % self.class_measure[
                        unique_class].degreeOfNonMembership)

                plt.plot(self.x_space, self.intersection_current_dominant_kernels[unique_class], color='black')
                plt.fill_between(self.x_space, self.intersection_current_dominant_kernels[unique_class], color='black',
                                 alpha=1.0,
                                 label='Uncertainty (%0.2f)' % self.class_measure[unique_class].degreeOfUncertainty)

            plt.legend()
            if isinstance(self.x_space[0], str) == True:
                plt.xlabel('Class')
            else:
                plt.xlabel('Score')
            plt.ylabel('Density')
            if title:
                plt.title(f'G measure presentation for class {unique_class}')
            plt.savefig("plots/" + file_name + "_class_" + str(unique_class) + ".png")
            plt.close()

        for i, (class_index, kernel) in enumerate(self.kernels.items()):
            color = colors[i]
            if isinstance(self.x_space[0], str) == True:
                plt.bar(self.x_space, kernel, width=bar_width, label=f'Class {class_index}',
                        color=color, alpha=0.5, edgecolor=color, linewidth=3.0)
            else:
                plt.plot(self.x_space, kernel, label=f'Class {class_index}', color=color)
                plt.fill_between(self.x_space, kernel, color=color, alpha=0.5)

        plt.legend()
        if isinstance(self.x_space[0], str) == True:
            plt.xlabel('Class')
        else:
            plt.xlabel('Score')
        plt.ylabel('Density')
        if title:
            plt.title('Class densities presentation')
        plt.savefig("plots/" + file_name + "_densities.png")
        plt.close()

    def plot_tresholds_chart(self, file_name='plot', title=True):
        if self.thresholds is not None and isinstance(self.x_space[0], str) == False:
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, len(self.unique_classes))]
            for i, (class_index, kernel) in enumerate(self.kernels.items()):
                color = colors[i]
                plt.plot(self.x_space, kernel, label=f'Class {class_index}', color=color)
                plt.fill_between(self.x_space, kernel, color=color, alpha=0.5)

            start = self.x_space[0]
            for class_index, threshold in self.thresholds:
                plt.axvline(x=threshold, color='black', linestyle='--', linewidth=1)
                if threshold == float('inf'):
                    threshold = self.x_space[-1]
                plt.text((start + threshold) / 2.0, plt.ylim()[1], f'Class {class_index}', rotation=0,
                         horizontalalignment='center', verticalalignment='top')
                start = threshold

            plt.legend(loc='center right')
            plt.xlabel('Score')
            plt.ylabel('Density')
            if title:
                plt.title('Class thresholds presentation')
            plt.savefig("plots/" + file_name + "_thresholds.png")
            plt.close()
        else:
            raise ValueError("G measure model not fitted from scores.")
