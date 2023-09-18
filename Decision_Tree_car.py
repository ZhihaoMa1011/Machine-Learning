from __future__ import absolute_import, division, print_function, \
                       unicode_literals

import csv
import sys
import numpy as np
from collections import defaultdict, Counter


class DecisionTree:
    class Node:
        def __init__(self, label, label_vals):
            # if this is leaf node (children is empty),
            # label = (predicted y value)
            self.label = label
            # key: unique label values, value: corresponding child node
            self.children = {}
            # special way to check if a numpy array is not None and not empty
            if label_vals is not None and label_vals.size > 0:
                for val in label_vals:
                    self.children[val] = None

        def __repr__(self):
            return "(" + ', '.join([self.label, str(self.children)]) + ")"

        def add_child(self, val, child):
            """
            Add a child node to this node.
            val: corresponding label value of this node
            child: child node
            """
            self.children[val] = child

    def __init__(self, train_input, test_input, max_depth, methods, train_output, test_output, metrics_out):
        # load train data
        train_labels, train_data = self.load(train_input)
        # obtain a list of unique y values; useful for printing y statistics later
        self.unique_y = np.unique(train_data[:, -1])
        # print general y statistics
        print(self.y_stats(train_data))
        # generate tree
        self.root = self.train_tree(train_labels, train_data, 0, max_depth, methods)
        # predict train data
        train_error = self.predict(train_labels, train_data, train_output, report_error=True)
        # predict test data
        test_labels, test_data = self.load(test_input)
        test_error = self.predict(test_labels, test_data, test_output, report_error=True)
        # output metrics
        self.write_metrics(train_error, test_error, metrics_out)

    def __repr__(self):
        return str(self.root)

    def count_unique(self, data):
        """
        A helper function for counting unique elements in data, as numpy 1.7.1
        doesn't support np.unique(data, return_counts=True).
        """
        counter_obj = Counter(data)
        return np.asarray(list(counter_obj.keys())), np.asarray(list(counter_obj.values()))

    def load(self, input_file):
        """
        Load data into memory.
        Returns: labels list, data table
        """
        # note that pandas is a better alternative to deal with tabular data
        # load raw data
        # note: there's a bug that prevents numpy from reading csv files like
        #       the following: (cannot correctly convert values to strings)
        # data = np.genfromtxt(input_file, delimiter=',', dtype=str,
        #                      autostrip=True, names=True)
        # print(data)
        # print(data.dtype.names)

        # raw_data = np.genfromtxt(input_file, delimiter=',', dtype=str,
        #                          autostrip=True)
        # # in numpy, arrays slices are just views on the original array
        # labels = raw_data[0]
        # data = raw_data[1:]

        raw_data = []
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                raw_data.append([item.strip() for item in row])

        labels = np.asarray(raw_data[0])
        data = np.asarray(raw_data[1:])

        return labels, data

    def predict(self, labels, data, output_file, report_error=False):
        """
        Predict using the trained tree, and output result to file.
        Returns prediction error if report_error=True (the last column of data
        must be the real y value).
        """
        total_errors = 0
        with open(output_file, 'w') as f:
            for record in data:
                pointer = self.root
                while pointer.children:
                    # get the value in record that fits into the current node
                    ind = np.where(labels == pointer.label)[0][0]
                    
                    if record[ind] in pointer.children:
                        pointer = pointer.children[record[ind]]
                    else:
                        # if the value doesn't exist in children, choose a default path
                        # here we choose the first child as default
                        pointer = list(pointer.children.values())[0]

                if report_error:
                    if pointer.label != record[-1]:
                        total_errors += 1
                f.write('%s\n' % pointer.label)

        if report_error:
            return total_errors / len(data)





    def write_metrics(self, train_error, test_error, output_file):
        """
        Write metrics to file.
        """
        with open(output_file, 'w') as f:
            f.write('error(train): %f\n' % train_error)
            f.write('error(test): %f\n' % test_error)
        print(train_error,  test_error)

    def train_tree(self, labels, data, depth, max_depth, method):
        """
        Main routine for training a decision tree.

        labels: a list of labels matching the dataset
        data: dataset (with the last column being the target column)
        max_depth: maximum depth of the trained tree
        """
        if labels.size <= 0 or data.size <= 0 or max_depth < 0:
            raise ValueError('Invalid argument(s) for tree training.')
            
        if labels.size == 1 or depth >= max_depth:
            # if there's only 1 column left or reached maximum depth, use
            # majority vote to create leaf node (note: not printed)
            return self.majority_vote(-1, data)

        label_ind, root = self.best_attrib(labels, data, depth, method)

        if label_ind >= 0:
            # while tree is not perfectly classified, divide dataset according
            # to each unique label value, and recursively build each subtree
            val_set = np.unique(data[:, label_ind])
            new_labels = np.delete(labels, label_ind)
            for val in val_set:  # for each subtree branch
                new_data = data[data[:, label_ind] == val]
                new_data = np.delete(new_data, label_ind, axis=1)
                # print this branch to console
                separator = ['| ' for i in range(depth + 1)]
                print('{}{} = {}: {}'.format(''.join(separator),
                                             labels[label_ind], val,
                                             self.y_stats(new_data)))
                child = self.train_tree(new_labels, new_data, depth + 1, max_depth, method)
                root.add_child(val, child)

        return root

    def best_attrib(self, labels, data, depth, method):

        if method == 1:
            root_entropy = self.entropy(-1, data)

            info_gains = []
            for i in range(len(labels) - 1):  # for each label (column) except y
                info_gains.append(self.info_gain(i, data, root_entropy))

            # pick the attribute with the largest info gain
            max_ind = np.argmax(info_gains)
            # -1 if perfectly classified (max info gain = 0)
            ind = -1 if info_gains[max_ind] <= 0 else max_ind
            new_node = None
            if ind >= 0:
                new_node = self.Node(labels[ind], np.unique(data[:, ind]))
            else:
                # tree is perfectly classified: create a leaf node instead
                # since y values are all the same, just use the first y value
                new_node = self.Node(data[0, -1], None)
            
            return ind, new_node
        

        if method == 2:
            root_majority_error= self.majority_error(-1, data)

            me_gains = []
            for i in range(len(labels) - 1):  # for each label (column) except y
                me_gains.append(self.me_gain(i, data, root_majority_error))

            # pick the attribute with the largest info gain
            max_ind = np.argmax(me_gains)
            # -1 if perfectly classified (max info gain = 0)
            ind = -1 if me_gains[max_ind] <= 0 else max_ind
            new_node = None
            if ind >= 0:
                new_node = self.Node(labels[ind], np.unique(data[:, ind]))
            else:
                # tree is perfectly classified: create a leaf node instead
                # since y values are all the same, just use the first y value
                new_node = self.Node(data[0, -1], None)
             
            return ind, new_node
        
        if method == 3:
            root_gini_index= self.gini_index(-1, data)

            gi_gains = []
            for i in range(len(labels) - 1):  # for each label (column) except y
                gi_gains.append(self.gi_gain(i, data, root_gini_index))

            # pick the attribute with the largest info gain
            max_ind = np.argmax(gi_gains)
            # -1 if perfectly classified (max info gain = 0)
            ind = -1 if gi_gains[max_ind] <= 0 else max_ind
            new_node = None
            if ind >= 0:
                new_node = self.Node(labels[ind], np.unique(data[:, ind]))
            else:
                # tree is perfectly classified: create a leaf node instead
                # since y values are all the same, just use the first y value
                new_node = self.Node(data[0, -1], None)

            return ind, new_node
        

    def info_gain(self, label_ind, data, root_entropy):
        """
        Calculate information gain (i.e. reduction in entropy) if the tree is
        split on the specified label.

        label_ind: column index representing the selected label in data
        data: dataset (with the last column being the target column)
        root_entropy: root entropy
        """
        # val_set, counts = np.unique(data[:, label_ind], return_counts=True)
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])

        # probabilities of each unique value P(x=x')
        val_freqs = counts / total_entries

        # specific conditional entropies H(Y|x=x')
        spec_cond_ents = []
        for val in val_set:
            spec_cond_ents.append(
                self.entropy(-1, data[data[:, label_ind] == val]))
        spec_cond_ents = np.asarray(spec_cond_ents)

        # conditional entropy H(Y|x)
        cond_ents = np.sum(val_freqs * spec_cond_ents)

        # info gain (mutual info)
        return root_entropy - cond_ents
    
    def me_gain(self, label_ind, data, root_majority_error):
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])
        # probabilities of each unique value P(x=x')
        val_freqs = counts / total_entries
        spec_cond_ents = []
        for val in val_set:
            spec_cond_ents.append(
                self.majority_error(-1, data[data[:, label_ind] == val]))
        spec_cond_ents = np.asarray(spec_cond_ents)

        cond_ents = np.sum(val_freqs * spec_cond_ents)
        
        return root_majority_error - cond_ents
    
    def gi_gain(self, label_ind, data, root_gini_index):
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])
        # probabilities of each unique value P(x=x')
        val_freqs = counts / total_entries
        spec_cond_ents = []
        for val in val_set:
            spec_cond_ents.append(
                self.gini_index(-1, data[data[:, label_ind] == val]))
        spec_cond_ents = np.asarray(spec_cond_ents)

        cond_ents = np.sum(val_freqs * spec_cond_ents)
        
        return root_gini_index - cond_ents

    def entropy(self, label_ind, data):
        """
        Calculate data entropy under the given label.

        label_ind: column index representing the selected label in data
        data: dataset (with the last column being the target column)
        """
        # val_set, counts = np.unique(data[:, label_ind], return_counts=True)
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])
        # probabilities of each unique value
        val_freqs = counts / total_entries
        # entropy in bits: use log base 2
        return (-val_freqs * np.log2(val_freqs)).sum()
    
    def majority_error(self, label_ind, data):
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])
        # probabilities of each unique value
        val_freqs = counts / total_entries
        
        return (1 - max(val_freqs))
    

    def gini_index(self, label_ind, data):
        val_set, counts = self.count_unique(data[:, label_ind])
        total_entries = len(data[:, label_ind])
        # probabilities of each unique value
        val_freqs = counts / total_entries
        return (1-sum((val_freqs)**2))


    def majority_vote(self, label_ind, data):
        """
        Returns the most frequent value under the specified label in the
        dataset.

        label_ind: column index representing the selected label in data
        data: dataset (with the last column being the target column)
        """
        # unique_vals, counts = np.unique(data[:, label_ind], return_counts=True)
        unique_vals, counts = self.count_unique(data[:, label_ind])
        # after transpose, val_counts is an array of arrays of this format:
        # [['value', 'count'], ...] (Note that count is converted to string
        # since numpy arrays must share the same dtype)
        val_counts = np.asarray((unique_vals, counts)).T
        # convert the second column to int before performing np.argsort(),
        # otherwise will sort alphanumerically
        sort_ind = np.argsort([int(x) for x in val_counts[:, -1]])
        # sort val_counts by 'counts' column (by numeric values)
        val_counts = val_counts[sort_ind]

        return self.Node(val_counts[-1, 0], None)

    def y_stats(self, data):
        """
        Returns a formatted string of y's statistics in data.
        E.g. "[14 A / 52 B / 0 C]"
        """
        # unique_vals, counts = np.unique(data[:, -1], return_counts=True)
        unique_vals, counts = self.count_unique(data[:, -1])
        val_counts = dict(zip(unique_vals, counts))
        unique_y_counts = []
        for y in self.unique_y:
            if y in val_counts:
                unique_y_counts.append(val_counts[y])
            else:
                unique_y_counts.append(0)

        # construct answer
        ans = []
        for i, y in enumerate(self.unique_y):
            ans.append('{} {}'.format(unique_y_counts[i], y))

        return '[' + ' / '.join(ans) + ']'



train_input = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\car\train.csv"
test_input = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\car\test.csv"
max_depth = 6
method = 3# information gain = 1 ; majority error = 2 ; gini index = 3; 
train_out = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\car\train_out.txt"
test_out = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\car\test_out.txt"
metrics_out = r"E:\U of U\OneDrive - University of Utah\Course\Machine Learning\car\metrics.txt"


tree = DecisionTree(train_input, test_input, max_depth, method, train_out, test_out, metrics_out)

